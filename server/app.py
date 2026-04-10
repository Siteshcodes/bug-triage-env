# server/app.py
import sys
import json
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from openenv.core.env_server import create_app
from model import TriageAction, TriageObservation
from environment import BugTriageEnvironment
from task import sample_bug, grade_action, TASKS
from fastapi import Response, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = create_app(
    BugTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="bug-triage-env",
)

TASKS_META = [
    {
        "id": "easy",
        "name": "Priority Assignment",
        "description": "Assign correct P0-P3 priority to a bug report",
        "difficulty": "easy",
        "grader": "server.task:priority_match",
        "reward_range": [0.0, 1.0]
    },
    {
        "id": "medium",
        "name": "Priority Labels and Team",
        "description": "Assign correct priority, labels, and team routing",
        "difficulty": "medium",
        "grader": "server.task:priority_label_team",
        "reward_range": [0.0, 1.0]
    },
    {
        "id": "hard",
        "name": "Full Triage",
        "description": "Full triage with priority, labels, team, milestone and security penalty",
        "difficulty": "hard",
        "grader": "server.task:full_triage",
        "reward_range": [0.0, 1.0]
    }
]

# ─────────────────────────────────────────────
# GLOBAL STATEFUL ENVIRONMENT
# The OpenEnv create_app() creates stateless endpoints that spin up
# a new environment per request. This breaks our reset→step flow
# because step() needs the bug from reset().
# We maintain a shared global instance to fix this.
# ─────────────────────────────────────────────
_global_env = BugTriageEnvironment()


# ─────────────────────────────────────────────
# REMOVE the framework's stateless /reset and /step routes,
# then add our own stateful versions.
# ─────────────────────────────────────────────
# Remove existing /reset and /step routes registered by create_app()
routes_to_remove = []
for route in app.routes:
    if hasattr(route, "path") and route.path in ("/reset", "/step", "/state"):
        routes_to_remove.append(route)
for route in routes_to_remove:
    app.routes.remove(route)


@app.get("/")
def root():
    return {"status": "ok", "env": "bug-triage-env"}

@app.get("/tasks")
def list_tasks():
    return TASKS_META

@app.get("/tasks/easy")
def task_easy():
    return TASKS_META[0]

@app.get("/tasks/medium")
def task_medium():
    return TASKS_META[1]

@app.get("/tasks/hard")
def task_hard():
    return TASKS_META[2]


# ─────────────────────────────────────────────
# CUSTOM STATEFUL /reset and /step endpoints
# ─────────────────────────────────────────────

@app.post("/reset")
async def custom_reset(request: Request):
    """Stateful reset — remembers the bug for the subsequent step() call."""
    global _global_env

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    task_id = body.get("task_id", "easy")
    seed = body.get("seed", None)
    episode_id = body.get("episode_id", None)

    _global_env = BugTriageEnvironment()
    obs = _global_env.reset(task_id=task_id, seed=seed, episode_id=episode_id)

    try:
        obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    except AttributeError:
        obs_dict = obs.dict()
        obs_dict.pop("reward", None)
        obs_dict.pop("done", None)
        obs_dict.pop("metadata", None)

    return {
        "observation": obs_dict,
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
async def custom_step(request: Request):
    """Stateful step — uses the bug from the last reset() call."""
    global _global_env

    body = await request.json()
    action_data = body.get("action", body)

    action = TriageAction(
        priority=action_data.get("priority", "P2"),
        labels=action_data.get("labels", ["bug"]),
        assigned_team=action_data.get("assigned_team", "backend"),
        milestone=action_data.get("milestone", "backlog"),
        reasoning=action_data.get("reasoning", ""),
    )

    obs = _global_env.step(action)

    try:
        obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    except AttributeError:
        obs_dict = obs.dict()
        obs_dict.pop("reward", None)
        obs_dict.pop("done", None)
        obs_dict.pop("metadata", None)

    reward = float(obs.reward) if obs.reward is not None else 0.05
    # Strictly clamp to open interval (0, 1)
    reward = max(0.01, min(0.99, reward))

    return {
        "observation": obs_dict,
        "reward": reward,
        "done": obs.done,
    }


@app.get("/state")
def custom_state():
    """Return current environment state."""
    global _global_env
    state = _global_env.get_state()
    try:
        return state.model_dump()
    except AttributeError:
        return state.dict()


# ─────────────────────────────────────────────
# Per-task reset shortcuts (convenience)
# ─────────────────────────────────────────────

@app.post("/tasks/easy/reset")
def reset_easy():
    global _global_env
    _global_env = BugTriageEnvironment()
    obs = _global_env.reset(task_id="easy")
    return {"task_id": "easy", "bug_report": obs.bug_report.model_dump(), "done": False, "reward": 0.05}

@app.post("/tasks/medium/reset")
def reset_medium():
    global _global_env
    _global_env = BugTriageEnvironment()
    obs = _global_env.reset(task_id="medium")
    return {"task_id": "medium", "bug_report": obs.bug_report.model_dump(), "done": False, "reward": 0.05}

@app.post("/tasks/hard/reset")
def reset_hard():
    global _global_env
    _global_env = BugTriageEnvironment()
    obs = _global_env.reset(task_id="hard")
    return {"task_id": "hard", "bug_report": obs.bug_report.model_dump(), "done": False, "reward": 0.05}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()