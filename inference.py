"""
inference.py — Bug Triage Env
OpenEnv Hackathon submission inference script.

Required env vars:
    API_BASE_URL   LLM endpoint (default: HuggingFace router)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key
    ENV_BASE_URL   Bug Triage env URL (default: HF Space)
"""

import os
import json
import time
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import BugTriageClient
from model import TriageAction

# ── config ──────────────────────────────────────────────────────────────
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
API_KEY       = os.getenv("HF_TOKEN") 
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL  = os.getenv("ENV_BASE_URL", "https://siteshcodes-bug-triage-env.hf.space")

TASK_NAME     = "bug-triage"
BENCHMARK     = "bug-triage-env"
MAX_STEPS     = 3
TEMPERATURE   = 0.0
MAX_TOKENS    = 400
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


# ── logging helpers ──────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── model call ───────────────────────────────────────────────────────────

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

        # strip accidental markdown fences
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
        print(f"[DEBUG] Model call failed: {exc}", flush=True)
        # fallback action
        return TriageAction(
            priority="P2",
            labels=["bug"],
            assigned_team="backend",
            milestone="backlog",
            reasoning="fallback due to model error",
        )


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    score = 0.0          
    success = False
    steps_taken = 0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        with BugTriageClient(base_url=ENV_BASE_URL) as env:

            task_order = ["easy", "medium", "hard"]

            for step_count, task_name in enumerate(task_order, start=1):

                obs = env.reset()

                bug_text = format_bug(obs)
                action = call_model(client, bug_text)

                result = env.step(action)

                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken = step_count

                action_str = (
                    f"priority={action.priority},"
                    f"team={action.assigned_team},"
                    f"milestone={action.milestone}"
                )

                log_step(
                    step=step_count,
                    action=action_str,
                    reward=reward,
                    done=True,
                    error=None,
                )

                time.sleep(0.5)

        score = sum(rewards) / 3 if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken + 1, action="none", reward=0.0, done=True, error=str(e))
        score = 0.0
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    main()