"""
inference.py — Bug Triage Env
OpenEnv Hackathon submission inference script.

Required env vars:
    API_BASE_URL   LiteLLM proxy base URL (injected by validator)
    API_KEY        API key (injected by validator)
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

_raw_base_url = os.getenv("API_BASE_URL", "").strip()
if not _raw_base_url:
    raise RuntimeError("API_BASE_URL is not set — validator must inject this")

# LiteLLM strictly requires /v1 at the end of the base URL
if not _raw_base_url.rstrip("/").endswith("/v1"):
    API_BASE_URL = _raw_base_url.rstrip("/") + "/v1"
else:
    API_BASE_URL = _raw_base_url.rstrip("/")

API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
)
if not API_KEY:
    raise RuntimeError("API_KEY is not set — validator must inject this")

MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://siteshcodes-bug-triage-env.hf.space")

TASK_IDS                = ["easy", "medium", "hard"]
BENCHMARK               = "bug-triage-env"
TEMPERATURE             = 0.0
MAX_TOKENS              = 400
SUCCESS_SCORE_THRESHOLD = 0.4

print(f"[CONFIG] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[CONFIG] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[CONFIG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)
print(f"[CONFIG] API_KEY={'set' if API_KEY else 'MISSING'}", flush=True)

# ── prompt ────────────────────────────────────────────────────────────────

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


def log_step(
    step: int,
    task_id: str,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    print(
        f"[STEP] step={step} task={task_id} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_task_end(task_id: str, reward: float) -> None:
    print(f"[TASK_END] task={task_id} reward={reward:.3f}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────

def format_bug(obs) -> str:
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

    # Strip markdown fences if model ignores instructions
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:].strip()

    data = json.loads(raw)

    action = TriageAction(
        priority=data.get("priority", "P2"),
        labels=data.get("labels", ["bug"]),
        assigned_team=data.get("assigned_team", "backend"),
        milestone=data.get("milestone", "backlog"),
        reasoning=data.get("reasoning", ""),
    )

    print(
        f"[LLM] Parsed action: priority={action.priority} "
        f"team={action.assigned_team} milestone={action.milestone}",
        flush=True,
    )
    return action


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards: List[float] = []
    score   = 0.0
    success = False

    log_start(env=BENCHMARK, model=MODEL_NAME)

    try:
        with BugTriageClient(base_url=ENV_BASE_URL) as env:
            for step, task_id in enumerate(TASK_IDS, start=1):
                log_task_start(task_id)

                obs    = env.reset(task_id)
                action = call_model(client, format_bug(obs))
                result = env.step(action)
                reward = float(result.reward or 0.0)
                rewards.append(reward)

                action_str = (
                    f"priority={action.priority},"
                    f"team={action.assigned_team},"
                    f"milestone={action.milestone}"
                )
                log_step(
                    step=step,
                    task_id=task_id,
                    action=action_str,
                    reward=reward,
                    done=True,
                )
                log_task_end(task_id, reward)
                time.sleep(0.5)

        score   = sum(rewards) / len(TASK_IDS)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] Exception during run: {type(exc).__name__}: {exc}", flush=True)
        score   = sum(rewards) / len(TASK_IDS) if rewards else 0.0
        success = False
        raise exc  # CRITICAL: never fail silently — let validator see real error

    finally:
        log_end(success, len(rewards), score, rewards)


if __name__ == "__main__":
    main()