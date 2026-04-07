# server/app.py
import sys
import os
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from openenv.core.env_server import create_app
from model import TriageAction, TriageObservation
from environment import BugTriageEnvironment

app = create_app(
    BugTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="bug-triage-env",
)
@app.get("/")
def root():
    return {"status": "ok", "env": "bug-triage-env"}
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Assign a single P0-P3 priority to a bug report",
                "grader": "priority_match",
                "reward_range": [0.0, 1.0]
            },
            {
                "id": "medium", 
                "description": "Assign priority, labels, and team routing",
                "grader": "priority_label_team",
                "reward_range": [0.0, 1.0]
            },
            {
                "id": "hard",
                "description": "Full triage with security escalation penalty",
                "grader": "full_triage",
                "reward_range": [0.0, 1.0]
            }
        ]
    }

@app.get("/tasks/easy")
def task_easy():
    return {"id": "easy", "grader": "priority_match", "reward_range": [0.0, 1.0]}

@app.get("/tasks/medium")
def task_medium():
    return {"id": "medium", "grader": "priority_label_team", "reward_range": [0.0, 1.0]}

@app.get("/tasks/hard")
def task_hard():
    return {"id": "hard", "grader": "full_triage", "reward_range": [0.0, 1.0]}

if __name__ == "__main__":
    main()