#models.py
from dataclasses import dataclass, field
from typing import List, Optional
from openenv.core.env_server import Action, Observation
from openenv.core.env_server.types import State


@dataclass
class BugReport:
    """A single GitHub-style bug report."""
    id: str
    title: str
    body: str
    author: str
    labels_hint: List[str]   # existing labels on the issue (may be empty)
    comments: List[str]      # top comments for context


@dataclass
class TriageAction(Action):
    """What the agent submits as its triage decision."""
    priority: str            # "P0" | "P1" | "P2" | "P3"
    labels: List[str]        # e.g. ["bug", "performance"]
    assigned_team: str       # e.g. "backend", "frontend", "infra", "security"
    milestone: str           # e.g. "v2.1", "backlog", "hotfix"
    reasoning: str           # brief explanation (rewarded for quality in hard task)


@dataclass
class TriageObservation(Observation):
    """What the agent sees after each step."""
    bug_report: BugReport
    task_id: str             # "easy" | "medium" | "hard"
    score: float             # cumulative score so far (0.0–1.0)
    feedback: str            # human-readable feedback on last action
    done: bool
    reward: float


@dataclass
class TriageState(State):
    """Internal episode state."""
    episode_id: str = ""
    current_task: str = "easy"
    step_count: int = 0
    total_score: float = 0.0
    tasks_completed: List[str] = field(default_factory=list)