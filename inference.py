"""
inference.py — Bug Triage Env
OpenEnv Hackathon submission inference script.

Required env vars:
    API_BASE_URL   LiteLLM proxy base URL (injected by validator)
    HF_TOKEN       API key (injected by validator)
    ENV_BASE_URL   Bug Triage env URL (optional)
    MODEL_NAME     Model identifier (optional)
"""

import os
import json
import time
import textwrap
import requests
from typing import List, Optional

from openai import OpenAI
from model import TriageAction, TriageObservation, BugReport


# ---------------------------------------------------------------------------
#  CONFIG — uses env vars required by hackathon spec
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "https://siteshcodes-bug-triage-env.hf.space"

if not API_KEY:
    raise RuntimeError("HF_TOKEN is not set")

TASK_IDS                = ["easy", "medium", "hard"]
BENCHMARK               = "bug-triage-env"
TEMPERATURE             = 0.0
MAX_TOKENS              = 500
MAX_STEPS               = 4       # Max steps per task (investigate + submit)
MAX_TOTAL_REWARD        = 1.0
SUCCESS_SCORE_THRESHOLD = 0.4

print(f"[CONFIG] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[CONFIG] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[CONFIG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)
print(f"[CONFIG] API_KEY={'set' if API_KEY else 'MISSING'}", flush=True)


# ---------------------------------------------------------------------------
#  INLINED CLIENT — self-contained, no external dependency
# ---------------------------------------------------------------------------

def _parse_observation(data: dict) -> TriageObservation:
    try:
        bug = BugReport.model_validate(data["bug_report"])
    except Exception:
        bug = BugReport(**data["bug_report"])
    return TriageObservation(
        bug_report=bug,
        task_id=data.get("task_id", "easy"),
        score=data.get("score", 0.0),
        feedback=data.get("feedback", ""),
        done=data.get("done", False),
        reward=data.get("reward", 0.0),
        body_visible=data.get("body_visible", False),
        comments_visible=data.get("comments_visible", False),
        logs_visible=data.get("logs_visible", False),
        similar_visible=data.get("similar_visible", False),
        steps_taken=data.get("steps_taken", 0),
        max_steps=data.get("max_steps", 6),
    )


class StepResult:
    def __init__(self, observation: TriageObservation, reward: float,
                 done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class BugTriageClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or ENV_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._session_id: Optional[str] = None

    def reset(self, task_id: str = "easy") -> TriageObservation:
        print(f"[ENV] Resetting env for task={task_id}", flush=True)
        payload = {"task_id": task_id}
        if self._session_id:
            payload["session_id"] = self._session_id

        response = self.session.post(
            f"{self.base_url}/reset", json=payload, timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self._session_id = data.get("session_id")
        return _parse_observation(data.get("observation", data))

    def step(self, action: TriageAction) -> StepResult:
        print(f"[ENV] Sending step: action_type={action.action_type}", flush=True)
        try:
            action_dict = action.model_dump()
        except AttributeError:
            action_dict = action.dict()

        payload = {"action": action_dict}
        if self._session_id:
            payload["session_id"] = self._session_id

        response = self.session.post(
            f"{self.base_url}/step", json=payload, timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        obs = _parse_observation(data.get("observation", data))

        reward = data.get("reward", obs.reward)
        if reward is None:
            reward = 0.0
        reward = float(reward)
        if obs.done:
            reward = max(0.01, min(0.99, reward))

        if "session_id" in data:
            self._session_id = data["session_id"]

        return StepResult(
            observation=obs, reward=reward,
            done=data.get("done", obs.done), info={},
        )

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
#  LLM PROMPTS
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior software engineering manager triaging a bug report.
    You will receive a bug report (possibly with partial information).
    Respond ONLY with valid JSON — no markdown, no explanation, no backticks.

    Return exactly this structure:
    {
      "priority": "P0",
      "labels": ["bug"],
      "assigned_team": "backend",
      "milestone": "hotfix",
      "reasoning": "one sentence explaining your decision"
    }

    Priority guide:
      P0 — production down, data loss, security vulnerability, 100% user impact
      P1 — major feature broken, significant user impact, no workaround
      P2 — degraded experience, workaround exists
      P3 — minor, cosmetic, docs, low impact

    Teams: backend | frontend | infra | security | devx
    Milestones: hotfix | v2.1 | backlog

    Important: Pay attention to security signals (SQL injection, XSS, auth bypass,
    data exposure). Security bugs should almost always be P0 + security team + hotfix.
""").strip()

INVESTIGATION_PROMPT = textwrap.dedent("""
    You are deciding whether to investigate further or submit your triage.
    You have seen the following information about a bug. Based on what you see,
    decide if you need more information or can triage now.

    Respond with ONLY one of these JSON formats:

    To investigate: {"action": "read_body"} or {"action": "read_comments"} or {"action": "check_logs"}
    To submit:
    {
      "action": "submit",
      "priority": "P0",
      "labels": ["bug"],
      "assigned_team": "backend",
      "milestone": "hotfix",
      "reasoning": "explanation"
    }

    Only investigate if the title and preview are genuinely ambiguous.
    If the bug is clearly a typo or clearly critical, submit immediately.
""").strip()


# ---------------------------------------------------------------------------
#  STRUCTURED LOGGING — strict [START]/[STEP]/[END] format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
#  BUG FORMATTING
# ---------------------------------------------------------------------------

def format_bug(obs: TriageObservation) -> str:
    """Format a bug observation into text the LLM can read."""
    bug = obs.bug_report
    parts = [f"Title: {bug.title}"]

    parts.append(f"\nDescription:\n{bug.body}")

    if obs.comments_visible and bug.comments:
        comments = "\n".join(f"  - {c}" for c in bug.comments)
        parts.append(f"\nComments:\n{comments}")

    if bug.labels_hint:
        parts.append(f"\nExisting labels: {', '.join(bug.labels_hint)}")

    if obs.logs_visible:
        if bug.stack_trace:
            parts.append(f"\nStack trace: {bug.stack_trace}")
        if bug.affected_component:
            parts.append(f"\nAffected component: {bug.affected_component}")
        if bug.severity_signals:
            parts.append(f"\nSeverity signals: {', '.join(bug.severity_signals)}")

    if obs.similar_visible and bug.related_bugs:
        parts.append(f"\nRelated bugs: {', '.join(bug.related_bugs)}")

    # Add visibility context
    visibility = []
    if not obs.body_visible:
        visibility.append("body (truncated)")
    if not obs.comments_visible:
        visibility.append("comments (hidden)")
    if not obs.logs_visible:
        visibility.append("logs (hidden)")
    if visibility:
        parts.append(f"\n[Hidden info: {', '.join(visibility)}]")

    parts.append(f"\nSteps used: {obs.steps_taken}/{obs.max_steps}")

    return "\n".join(parts)


def format_bug_for_decision(obs: TriageObservation) -> str:
    """Shorter format for the investigation decision."""
    bug = obs.bug_report
    text = f"Title: {bug.title}\nPreview: {bug.body[:150]}"
    if obs.body_visible:
        text += f"\n\nFull body visible."
    if obs.comments_visible and bug.comments:
        text += f"\nComments: {len(bug.comments)} visible."
    text += f"\nSteps remaining: {obs.max_steps - obs.steps_taken}"
    return text


# ---------------------------------------------------------------------------
#  MODEL CALLS
# ---------------------------------------------------------------------------

def decide_action(client: OpenAI, obs: TriageObservation) -> dict:
    """Ask the LLM whether to investigate or submit."""
    bug_text = format_bug_for_decision(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": INVESTIGATION_PROMPT},
                {"role": "user", "content": bug_text},
            ],
            temperature=TEMPERATURE,
            max_tokens=200,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[DEBUG] Decision model call failed: {e}", flush=True)
        return {"action": "submit"}


def call_model(client: OpenAI, bug_text: str) -> TriageAction:
    """Ask the LLM to triage the bug report."""
    print("[LLM] Sending triage request to model...", flush=True)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": bug_text},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )

    raw = (completion.choices[0].message.content or "").strip()
    print(f"[LLM] Raw response: {raw[:200]}", flush=True)

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[LLM] JSON parse failed: {e}. Using defaults.", flush=True)
        data = {}

    action = TriageAction(
        action_type="submit",
        priority=data.get("priority", "P2"),
        labels=data.get("labels", ["bug"]),
        assigned_team=data.get("assigned_team", "backend"),
        milestone=data.get("milestone", "backlog"),
        reasoning=data.get("reasoning", ""),
    )

    print(
        f"[LLM] Parsed: priority={action.priority} "
        f"team={action.assigned_team} milestone={action.milestone}",
        flush=True,
    )
    return action


# ---------------------------------------------------------------------------
#  MAIN — multi-step agent with per-task [START]/[STEP]/[END] logging
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []

    with BugTriageClient(base_url=ENV_BASE_URL) as env:
        for task_id in TASK_IDS:
            rewards: List[float] = []
            score = 0.0
            success = False
            steps_taken = 0

            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            try:
                obs = env.reset(task_id=task_id)

                for step_num in range(1, MAX_STEPS + 1):
                    if obs.done:
                        break

                    # Decide: investigate or submit?
                    # For efficiency, check if we have enough info
                    # On step 1, always read full body; on later steps, decide
                    if step_num == 1 and not obs.body_visible:
                        # First step: read the full body
                        action = TriageAction(action_type="read_body")
                        result = env.step(action)
                        obs = result.observation
                        steps_taken = step_num

                        log_step(
                            step=step_num,
                            action="investigate:read_body",
                            reward=0.0,
                            done=result.done,
                        )

                        if result.done:
                            rewards.append(result.reward)
                            break
                        continue

                    elif step_num == 2 and not obs.comments_visible:
                        # Second step: read comments for extra context
                        action = TriageAction(action_type="read_comments")
                        result = env.step(action)
                        obs = result.observation
                        steps_taken = step_num

                        log_step(
                            step=step_num,
                            action="investigate:read_comments",
                            reward=0.0,
                            done=result.done,
                        )

                        if result.done:
                            rewards.append(result.reward)
                            break
                        continue

                    # Now submit the triage decision
                    bug_text = format_bug(obs)
                    action = call_model(client, bug_text)
                    result = env.step(action)
                    obs = result.observation
                    steps_taken = step_num

                    reward = float(result.reward or 0.0)
                    if result.done:
                        reward = max(0.01, min(0.99, reward))
                    rewards.append(reward)

                    action_str = (
                        f"priority={action.priority},"
                        f"team={action.assigned_team},"
                        f"milestone={action.milestone}"
                    )

                    log_step(
                        step=step_num,
                        action=action_str,
                        reward=reward,
                        done=result.done,
                    )

                    if result.done:
                        break

                # Calculate score
                if rewards:
                    score = sum(rewards) / MAX_TOTAL_REWARD
                else:
                    score = 0.0
                score = min(max(score, 0.01), 0.99)
                success = score >= SUCCESS_SCORE_THRESHOLD

            except Exception as exc:
                print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
                score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.05
                score = min(max(score, 0.01), 0.99)
                success = False

            log_end(success, steps_taken, score, rewards)
            all_scores.append(score)

            time.sleep(0.5)

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"[SUMMARY] tasks={len(all_scores)} avg_score={avg_score:.2f} "
        f"scores={all_scores}",
        flush=True,
    )


if __name__ == "__main__":
    main()