# server/app.py
import sys
import os
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from openenv.core.env_server import create_app
from model import TriageAction, TriageObservation
from environment import BugTriageEnvironment, SessionManager, TASKS_META
from task import sample_bug, grade_action, TASKS
from fastapi import Response, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any

app = create_app(
    BugTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="bug-triage-env",
)

# Session manager replaces the broken global state
sessions = SessionManager(max_sessions=500, ttl_seconds=600)

# Fallback env for backward-compatible (non-session) requests
_fallback_env = BugTriageEnvironment()
_fallback_answer = None


# Remove default routes from create_app — we override them
routes_to_remove = []
for route in app.routes:
    if hasattr(route, "path") and route.path in ("/reset", "/step", "/state"):
        routes_to_remove.append(route)
for route in routes_to_remove:
    app.routes.remove(route)


# ---------------------------------------------------------------------------
#  CORE ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "env": "bug-triage-env",
        "version": "2.0.0",
        "active_sessions": sessions.active_count,
    }


@app.get("/")
def root():
    """Serve the interactive demo frontend."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Bug Triage Environment v2.0.0", "docs": "/docs"}


@app.get("/web")
def web_ui():
    """Alias for the frontend."""
    return root()


@app.get("/tasks")
def list_tasks():
    return TASKS_META


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    for t in TASKS_META:
        if t["id"] == task_id:
            return t
    raise HTTPException(404, detail={
        "error": "task_not_found",
        "message": f"Task '{task_id}' not found. Valid: easy, medium, hard",
    })


# ---------------------------------------------------------------------------
#  SESSION-BASED RESET / STEP / STATE
# ---------------------------------------------------------------------------

@app.post("/reset")
async def custom_reset(request: Request):
    """Start a new episode. Returns a session_id for subsequent step() calls."""
    global _fallback_env, _fallback_answer

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    task_id = body.get("task_id", "easy")
    seed = body.get("seed", None)
    episode_id = body.get("episode_id", None)
    session_id = body.get("session_id", None)

    # If session_id provided, reuse that session
    if session_id:
        env = sessions.get_session(session_id)
        if env is None:
            session_id, env = sessions.create_session()
    else:
        session_id, env = sessions.create_session()

    obs = env.reset(task_id=task_id, seed=seed, episode_id=episode_id)

    # Also update fallback for backward compatibility
    _fallback_env = env

    try:
        obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    except AttributeError:
        obs_dict = obs.dict()
        obs_dict.pop("reward", None)
        obs_dict.pop("done", None)
        obs_dict.pop("metadata", None)

    return {
        "session_id": session_id,
        "observation": obs_dict,
        "reward": 0.0,
        "done": False,
    }


@app.post("/step")
async def custom_step(request: Request):
    """Process an action — either investigation or final triage submission."""
    global _fallback_env

    body = await request.json()
    action_data = body.get("action", body)
    session_id = body.get("session_id", None)

    # Find the right environment
    env = None
    if session_id:
        env = sessions.get_session(session_id)
    if env is None:
        env = _fallback_env

    action = TriageAction(
        action_type=action_data.get("action_type", "submit"),
        priority=action_data.get("priority", "P2"),
        labels=action_data.get("labels", ["bug"]),
        assigned_team=action_data.get("assigned_team", "backend"),
        milestone=action_data.get("milestone", "backlog"),
        reasoning=action_data.get("reasoning", ""),
    )

    obs = env.step(action)

    try:
        obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    except AttributeError:
        obs_dict = obs.dict()
        obs_dict.pop("reward", None)
        obs_dict.pop("done", None)
        obs_dict.pop("metadata", None)

    reward = float(obs.reward) if obs.reward is not None else 0.0
    reward = max(0.01, min(0.99, reward)) if obs.done else 0.0

    response_data = {
        "observation": obs_dict,
        "reward": reward,
        "done": obs.done,
    }

    if session_id:
        response_data["session_id"] = session_id

    # Cleanup session when episode is done
    if obs.done and session_id:
        sessions.remove_session(session_id)

    return response_data


@app.get("/state")
def custom_state(session_id: Optional[str] = None):
    """Return current environment state."""
    env = None
    if session_id:
        env = sessions.get_session(session_id)
    if env is None:
        env = _fallback_env

    state = env.get_state()
    try:
        return state.model_dump()
    except AttributeError:
        return state.dict()


# ---------------------------------------------------------------------------
#  PER-TASK SHORTCUT ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/tasks/easy/reset")
async def reset_easy():
    session_id, env = sessions.create_session()
    obs = env.reset(task_id="easy")
    return {
        "session_id": session_id,
        "task_id": "easy",
        "bug_report": obs.bug_report.model_dump(),
        "done": False,
        "reward": 0.0,
    }


@app.post("/tasks/medium/reset")
async def reset_medium():
    session_id, env = sessions.create_session()
    obs = env.reset(task_id="medium")
    return {
        "session_id": session_id,
        "task_id": "medium",
        "bug_report": obs.bug_report.model_dump(),
        "done": False,
        "reward": 0.0,
    }


@app.post("/tasks/hard/reset")
async def reset_hard():
    session_id, env = sessions.create_session()
    obs = env.reset(task_id="hard")
    return {
        "session_id": session_id,
        "task_id": "hard",
        "bug_report": obs.bug_report.model_dump(),
        "done": False,
        "reward": 0.0,
    }


# ---------------------------------------------------------------------------
#  LEADERBOARD
# ---------------------------------------------------------------------------

_leaderboard = []


@app.get("/leaderboard")
def get_leaderboard():
    """Return top 50 agent scores."""
    return sorted(_leaderboard, key=lambda x: x.get("avg_score", 0), reverse=True)[:50]


@app.post("/leaderboard/submit")
async def submit_to_leaderboard(request: Request):
    """Submit agent scores to the leaderboard."""
    body = await request.json()
    entry = {
        "agent_name": body.get("agent_name", "anonymous"),
        "model": body.get("model", "unknown"),
        "scores": body.get("scores", {}),
        "avg_score": body.get("avg_score", 0.0),
    }
    _leaderboard.append(entry)
    rank = sorted(
        _leaderboard, key=lambda x: x.get("avg_score", 0), reverse=True
    ).index(entry) + 1
    return {"status": "submitted", "rank": rank, "total_entries": len(_leaderboard)}


# ---------------------------------------------------------------------------
#  ENTRYPOINT
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()