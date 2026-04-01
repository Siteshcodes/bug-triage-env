# baseline.py
# Runs a Groq-hosted LLaMA model against all 3 tasks
# Set env vars: GROQ_API_KEY, ENV_BASE_URL (optional)

import os
import json
from groq import Groq
from client import BugTriageClient
from model import TriageAction

# ── config ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"   # strong + free on Groq
TEMPERATURE = 0.0
MAX_TOKENS = 400

SYSTEM_PROMPT = """You are a senior software engineering manager.
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
Milestones: hotfix | v2.1 | backlog"""


def format_bug(obs) -> str:
    bug = obs.bug_report
    return (
        f"Title: {bug.title}\n\n"
        f"Description:\n{bug.body}\n\n"
        f"Existing labels: {', '.join(bug.labels_hint) or 'none'}\n"
        f"Comments:\n" + "\n".join(f"  - {c}" for c in bug.comments)
    )


def call_model(groq_client: Groq, bug_text: str) -> TriageAction:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": bug_text},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = response.choices[0].message.content.strip()

    # strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return TriageAction(
        priority=data["priority"],
        labels=data.get("labels", []),
        assigned_team=data.get("assigned_team", "backend"),
        milestone=data.get("milestone", "backlog"),
        reasoning=data.get("reasoning", ""),
    )


def main():
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set. Get a free key at console.groq.com")

    groq_client = Groq(api_key=GROQ_API_KEY)
    scores = {}
    step_count = 0

    print("=" * 50)
    print("  Bug Triage Env — Baseline Inference Script")
    print(f"  Model: {MODEL}")
    print("=" * 50)

    with BugTriageClient() as env:
        obs = env.reset()

        while not obs.done:
            task = obs.task_id
            print(f"\n── Task: {task.upper()} ──")
            print(f"  Bug: {obs.bug_report.title}")

            bug_text = format_bug(obs)
            action = call_model(groq_client, bug_text)

            print(f"  → Priority:  {action.priority}")
            print(f"  → Labels:    {action.labels}")
            print(f"  → Team:      {action.assigned_team}")
            print(f"  → Milestone: {action.milestone}")

            result = env.step(action)
            obs = result.observation

            print(f"  ✓ Reward:    {result.reward:.3f}")
            print(f"  ✓ Feedback:  {obs.feedback}")

            scores[task] = result.reward
            step_count += 1

    print("\n" + "=" * 50)
    print("  BASELINE SCORES")
    print("=" * 50)
    task_order = ["easy", "medium", "hard"]
    total = 0.0
    for task in task_order:
        s = scores.get(task, 0.0)
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {task:<8} {bar}  {s:.3f}")
        total += s
    avg = total / max(step_count, 1)
    print(f"\n  Average score: {avg:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()