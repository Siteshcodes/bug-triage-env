"""
inference.py — Bug Triage Env
OpenEnv Hackathon submission inference script.

Required env vars:
    HF_TOKEN       HuggingFace API key
    ENV_BASE_URL   Bug Triage env URL (optional)
    MODEL_NAME     Model identifier (optional)
"""

import os
import json
import time
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import BugTriageClient
from model import TriageAction

# ── config ───────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://siteshcodes-bug-triage-env.hf.space")

TASK_IDS                = ["easy", "medium", "hard"]
BENCHMARK               = "bug-triage-env"
TEMPERATURE             = 0.0
MAX_TOKENS              = 400
SUCCESS_SCORE_THRESHOLD = 0.4

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


# ── logging ───────────────────────────────────────────────────────────────

def log_start(env: str, model: str) -> None:
    print(f"[START] env={env} model={model}", flush=True)


def log_task_start(task_id: str) -> None:
    print(f"\n--- Starting Task: {task_id} ---", flush=True)


def log_step(step: int, task_id: str, action: str, reward: float,
             done: bool, error: Optional[str] = None) -> None:
    print(
        f"[STEP] step={step} task={task_id} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_task_end(task_id: str, reward: float) -> None:
    print(f"Task {task_id} completed. Final Reward: {reward:.3f}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── model ─────────────────────────────────────────────────────────────────

def format_bug(obs) -> str:
    bug = obs.bug_report
    comments = "\n".join(f"  - {c}" for c in bug.comments) or "  None"
    return (
        f"Title: {bug.title}\n\n"
        f"Description:\n{bug.body}\n\n"
        f"Existing labels: {', '.join(bug.labels_hint) or 'none'}\n"
        f"Comments:\n{comments}"
    )


def call_model(client: OpenAI, bug_text: str) -> TriageAction:
    try:
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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return TriageAction(
            priority=data.get("priority", "P2"),
            labels=data.get("labels", ["bug"]),
            assigned_team=data.get("assigned_team", "backend"),
            milestone=data.get("milestone", "backlog"),
            reasoning=data.get("reasoning", ""),
        )
    except Exception as exc:
        print(f"[DEBUG] model call failed: {exc}", flush=True)
        return TriageAction(
            priority="P2",
            labels=["bug"],
            assigned_team="backend",
            milestone="backlog",
            reasoning="fallback due to model error",
        )


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN not set.")

    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards: List[float] = []
    score   = 0.0
    success = False

    log_start(env=BENCHMARK, model=MODEL_NAME)

    try:
        with BugTriageClient(base_url=ENV_BASE_URL) as env:
            for step, task_id in enumerate(TASK_IDS, start=1):
                log_task_start(task_id)

                obs    = env.reset(task_id=task_id)
                action = call_model(client, format_bug(obs))
                result = env.step(action)
                reward = float(result.reward or 0.0)
                rewards.append(reward)

                action_str = (
                    f"priority={action.priority},"
                    f"team={action.assigned_team},"
                    f"milestone={action.milestone}"
                )
                log_step(step=step, task_id=task_id, action=action_str,
                         reward=reward, done=True)
                log_task_end(task_id, reward)
                time.sleep(0.5)

        score   = sum(rewards) / len(TASK_IDS)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        score   = sum(rewards) / len(TASK_IDS) if rewards else 0.0
        success = False

    finally:
        log_end(success, len(rewards), score, rewards)


if __name__ == "__main__":
    main()