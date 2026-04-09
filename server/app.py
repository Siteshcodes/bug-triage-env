# server/app.py
import sys
import json
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from openenv.core.env_server import create_app
from model import TriageAction, TriageObservation
from environment import BugTriageEnvironment
from task import sample_bug, grade_action, TASKS
from fastapi import Response
from pydantic import BaseModel
from typing import Optional

app = create_app(
    BugTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="bug-triage-env",
)

TASKS_META = [
    {
        "id": "easy",
        "description": "Assign correct P0-P3 priority to a bug report",
        "grader": "priority_match",
        "reward_range": [0.05, 0.95]
    },
    {
        "id": "medium",
        "description": "Assign correct priority, labels, and team routing",
        "grader": "priority_label_team",
        "reward_range": [0.05, 0.95]
    },
    {
        "id": "hard",
        "description": "Full triage with priority, labels, team, milestone and security penalty",
        "grader": "full_triage",
        "reward_range": [0.05, 0.95]
    }
]

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

@app.post("/tasks/easy/reset")
def reset_easy():
    bug = sample_bug("easy")
    return {"task_id": "easy", "bug_report": bug.dict(), "done": False, "reward": 0.05}

@app.post("/tasks/medium/reset")
def reset_medium():
    bug = sample_bug("medium")
    return {"task_id": "medium", "bug_report": bug.dict(), "done": False, "reward": 0.05}

@app.post("/tasks/hard/reset")
def reset_hard():
    bug = sample_bug("hard")
    return {"task_id": "hard", "bug_report": bug.dict(), "done": False, "reward": 0.05}
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()