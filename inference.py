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


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "https://siteshcodes-bug-triage-env.hf.space"

if not API_KEY:
    raise RuntimeError("HF_TOKEN is not set")

TASK_IDS                = ["easy", "medium", "hard"]
BENCHMARK               = "bug-triage-env"
TEMPERATURE             = 0.0
MAX_TOKENS              = 400
MAX_STEPS               = 1       # Each task is 1 step (reset → step → done)
MAX_TOTAL_REWARD        = 1.0     # Per-task max reward
SUCCESS_SCORE_THRESHOLD = 0.4

print(f"[CONFIG] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[CONFIG] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[CONFIG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)
print(f"[CONFIG] API_KEY={'set' if API_KEY else 'MISSING'}", flush=True)

#inlined client 

def _parse_observation(data: dict) -> TriageObservation:
    try:
        bug = BugReport.model_validate(data["bug_report"])
    except Exception:
        bug = BugReport(**data["bug_report"])
    return TriageObservation(
        bug_report=bug,
        task_id=data.get("task_id", "easy"),
        score=data.get("score", 0.05),
        feedback=data.get("feedback", ""),
        done=data.get("done", False),
        reward=data.get("reward", 0.05),
    )


class StepResult:
    def __init__(self, observation: TriageObservation, reward: float, done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class BugTriageClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or ENV_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self, task_id: str = "easy") -> TriageObservation:
        print(f"[ENV] Resetting env for task={task_id}", flush=True)
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return _parse_observation(data.get("observation", data))

    def step(self, action: TriageAction) -> StepResult:
        print("[ENV] Sending step action...", flush=True)
        try:
            action_dict = action.model_dump()
        except AttributeError:
            action_dict = action.dict()
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action_dict},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        obs = _parse_observation(data.get("observation", data))
        reward = data.get("reward", obs.reward)
        if reward is None or reward == 0:
            reward = 0.05
        reward = float(reward)
        # Strictly clamp to open interval (0, 1)
        reward = max(0.01, min(0.99, reward))
        return StepResult(
            observation=obs,
            reward=reward,
            done=data.get("done", obs.done),
            info={},
        )

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()




SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior software engineering manager.
    You will receive a bug report and must triage it. Respond ONLY with
    valid JSON — no markdown, no explanation, no backticks.

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
""").strip()




def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )




def format_bug(obs: TriageObservation) -> str:
    bug = obs.bug_report
    comments = "\n".join(f"  - {c}" for c in bug.comments) if bug.comments else "  None"
    return (
        f"Title: {bug.title}\n\n"
        f"Description:\n{bug.body}\n\n"
        f"Existing labels: {', '.join(bug.labels_hint) if bug.labels_hint else 'none'}\n"
        f"Comments:\n{comments}"
    )


def call_model(client: OpenAI, bug_text: str) -> TriageAction:
    print("[LLM] Sending request to model...", flush=True)

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




def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


    all_scores = []

    with BugTriageClient(base_url=ENV_BASE_URL) as env:
        for task_id in TASK_IDS:
            rewards: List[float] = []
            score = 0.0
            success = False
            steps_taken = 0

            # [START] for this task
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            try:
                obs = env.reset(task_id=task_id)
                action = call_model(client, format_bug(obs))
                result = env.step(action)

                reward = float(result.reward or 0.05)
                # Strictly clamp each reward to (0, 1) exclusive
                reward = max(0.01, min(0.99, reward))
                rewards.append(reward)
                steps_taken = 1

                action_str = (
                    f"priority={action.priority},"
                    f"team={action.assigned_team},"
                    f"milestone={action.milestone}"
                )
                log_step(
                    step=1,
                    action=action_str,
                    reward=reward,
                    done=True,
                )

                
                score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
                score = min(max(score, 0.01), 0.99)
                success = score >= SUCCESS_SCORE_THRESHOLD

            except Exception as exc:
                print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
                score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.05
                score = min(max(score, 0.01), 0.99)
                success = False

            # [END] for this task
            log_end(success, steps_taken, score, rewards)
            all_scores.append(score)

            time.sleep(0.5)

  
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[SUMMARY] tasks={len(all_scores)} avg_score={avg_score:.2f} scores={all_scores}", flush=True)


if __name__ == "__main__":
    main()