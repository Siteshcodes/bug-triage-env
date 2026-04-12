# model.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
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
    severity_signals: List[str] = Field(default_factory=list)
    related_bugs: List[str] = Field(default_factory=list)
    stack_trace: str = ""
    affected_component: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TriageAction(Action):
    """What the agent submits — either an investigation or a final triage decision."""
    action_type: str = "submit"     # "read_body" | "read_comments" | "check_logs" | "check_similar" | "submit"

    # Only used when action_type == "submit"
    priority: str = "P2"
    labels: List[str] = Field(default_factory=list)
    assigned_team: str = "backend"
    milestone: str = "backlog"
    reasoning: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TriageObservation(Observation):
    """What the agent sees after each step — progressively reveals info."""
    bug_report: BugReport
    task_id: str = "easy"
    score: float = 0.0
    feedback: str = ""
    done: bool = False
    reward: float = 0.0

    # Progressive visibility fields
    body_visible: bool = False
    comments_visible: bool = False
    logs_visible: bool = False
    similar_visible: bool = False
    steps_taken: int = 0
    max_steps: int = 6

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TriageState(State):
    """Internal episode state."""
    episode_id: str = ""
    session_id: str = ""
    current_task: str = "easy"
    step_count: int = 0
    total_score: float = 0.0
    tasks_completed: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)