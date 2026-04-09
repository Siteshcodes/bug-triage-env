# server/environment.py
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")
import uuid
from openenv.core.env_server.interfaces import Environment
from model import TriageAction, TriageObservation, TriageState, BugReport
from task import grade_action, sample_bug

VALID_TASKS = ["easy", "medium", "hard"]

TASKS_META = [
    {"id": "easy", "grader": "priority_match", "reward_range": [0.05, 0.95],
     "description": "Assign a single P0-P3 priority to a bug report"},
    {"id": "medium", "grader": "priority_label_team", "reward_range": [0.05, 0.95],
     "description": "Assign priority, labels, and team routing"},
    {"id": "hard", "grader": "full_triage", "reward_range": [0.05, 0.95],
     "description": "Full triage with security escalation penalty"},
]

class BugTriageEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._current_task_key: str = "easy"
        self._episode_done: bool = False
        self._current_bug: BugReport = sample_bug("easy")
        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            current_task="easy",
            step_count=0,
            total_score=0.0,
            tasks_completed=[],
        )

    def get_metadata(self):
        try:
            from openenv.core.env_server.types import EnvironmentMetadata
            return EnvironmentMetadata(
                name="bug-triage-env",
                description="Bug triage RL environment with 3 tasks of increasing difficulty",
                version="1.0.0",
                author="Siteshcodes",
                tasks=TASKS_META,
            )
        except Exception:
            return {
                "name": "bug-triage-env",
                "description": "Bug triage RL environment with 3 tasks of increasing difficulty",
                "version": "1.0.0",
                "author": "Siteshcodes",
                "tasks": TASKS_META,
            }

    def reset(self, task_id: str = "easy", seed: int = None, episode_id: str = None, **kwargs) -> TriageObservation:
        """Start a fresh episode for the specified task."""
        if task_id not in VALID_TASKS:
            task_id = "easy"

        self._current_task_key = task_id
        self._episode_done = False
        self._current_bug = sample_bug(task_id)
        self._state = TriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            current_task=task_id,
            step_count=0,
            total_score=0.0,
            tasks_completed=[],
        )
        return TriageObservation(
            bug_report=self._current_bug,
            task_id=task_id,
            score=0.0,
            feedback=f"Episode started for task: {task_id}. Triage this bug report.",
            done=False,
            reward=0.0,
        )

    def step(self, action: TriageAction) -> TriageObservation:
        """Process the agent's triage action — one step, then done."""
        if self._episode_done:
            return TriageObservation(
                bug_report=self._current_bug,
                task_id=self._current_task_key,
                score=0.0,
                feedback="Episode already complete. Call reset() to start a new episode.",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        task_key = self._current_task_key

        score, feedback = grade_action(task_key, self._current_bug, action)

        self._state.total_score += score
        self._state.tasks_completed.append(task_key)
        self._episode_done = True

        return TriageObservation(
            bug_report=self._current_bug,
            task_id=task_key,
            score=round(score, 3),
            feedback=feedback,
            done=True,
            reward=round(score, 3),
        )

    @property
    def state(self) -> TriageState:
        return self._state

    def get_state(self) -> TriageState:
        return self._state