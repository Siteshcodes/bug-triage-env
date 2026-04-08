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

# Remove /metadata and /reset routes registered by create_app so we can override them
app.routes[:] = [
    r for r in app.routes
    if not (hasattr(r, "path") and r.path in ["/metadata", "/reset"])
]

TASKS_META = [
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

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None

@app.get("/metadata")
def metadata():
    return Response(
        content=json.dumps({
            "name": "bug-triage-env",
            "description": "Bug triage RL environment with 3 tasks of increasing difficulty",
            "readme_content": None,
            "version": "1.0.0",
            "author": "Siteshcodes",
            "documentation_url": "https://siteshcodes-bug-triage-env.hf.space/docs",
            "tasks": TASKS_META
        }),
        media_type="application/json"
    )

@app.post("/reset")
def reset(request: ResetRequest = None):
    task_id = request.task_id if request else "easy"
    if task_id not in ["easy", "medium", "hard"]:
        task_id = "easy"
    env = BugTriageEnvironment()
    obs = env.reset(task_id=task_id)
    return obs

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
    return {"task_id": "easy", "bug_report": bug.dict(), "done": False, "reward": 0.0}

@app.post("/tasks/medium/reset")
def reset_medium():
    bug = sample_bug("medium")
    return {"task_id": "medium", "bug_report": bug.dict(), "done": False, "reward": 0.0}

@app.post("/tasks/hard/reset")
def reset_hard():
    bug = sample_bug("hard")
    return {"task_id": "hard", "bug_report": bug.dict(), "done": False, "reward": 0.0}

@app.post("/grader")
def grader_endpoint(task_id: str, action: TriageAction):
    bug = sample_bug(task_id)
    score, feedback = grade_action(task_id, bug, action)
    return {"task_id": task_id, "score": score, "feedback": feedback}

@app.get("/baseline")
def baseline():
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        bug = sample_bug(task_id)
        answer = TASKS[task_id]["answers"][bug.id]
        action = TriageAction(
            priority=answer["priority"],
            labels=answer.get("labels", []),
            assigned_team=answer.get("assigned_team", ""),
            milestone=answer.get("milestone", ""),
        )
        score, feedback = grade_action(task_id, bug, action)
        results[task_id] = {"score": score, "feedback": feedback}
    return results

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()