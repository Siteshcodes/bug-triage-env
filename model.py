# model.py
from typing import List
from pydantic import BaseModel, Field
from openenv.core.env_server import Action, Observation
from openenv.core.env_server.types import State





class BugReport(BaseModel):
    """A single GitHub-style bug report."""
    id: str
    title: str
    body: str
    author: str
    labels_hint: List[str] = Field(default_factory=list)
    comments: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True




class TriageAction(Action):
    """What the agent submits as its triage decision."""
    priority: str            # "P0" | "P1" | "P2" | "P3"
    labels: List[str] = Field(default_factory=list)
    assigned_team: str = "backend"
    milestone: str = "backlog"
    reasoning: str = ""

    class Config:
        arbitrary_types_allowed = True


class TriageObservation(Observation):
    """What the agent sees after each step."""
    bug_report: BugReport
    task_id: str = "easy"
    score: float = 0.0
    feedback: str = ""
    done: bool = False
    reward: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class TriageState(State):
    """Internal episode state."""
    episode_id: str = ""
    current_task: str = "easy"
    step_count: int = 0
    total_score: float = 0.0
    tasks_completed: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True