# server/environment.py
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

import uuid
import random
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from model import TriageAction, TriageObservation, TriageState, BugReport
from task import TASKS, grade_action
class BugTriageEnvironment(Environment):
    """
    Bug Report Triage RL environment.
    Agent reads GitHub-style bug reports and must triage them
    by assigning priority, labels, team, and milestone.
    """

    def __init__(self):
        self._state = TriageState(episode_id=str(uuid.uuid4()))
        self._current_bug: Optional[BugReport] = None
        self._current_task_key: str = "easy"

    def reset(self) -> TriageObservation:
        self._state = TriageState(episode_id=str(uuid.uuid4()))
        self._current_task_key = "easy"
        self._current_bug = random.choice(TASKS["easy"]["bugs"])

        return TriageObservation(
            bug_report=self._current_bug,
            task_id="easy",
            score=0.0,
            feedback="New episode started. Triage this bug report.",
            done=False,
            reward=0.0,
        )

    def step(self, action: TriageAction) -> TriageObservation:
        self._state.step_count += 1
        task_key = self._current_task_key

        # Grade the action
        assert self._current_bug is not None, "step() called before reset()"
        score, feedback = grade_action(task_key, self._current_bug, action)
        self._state.total_score += score

        # Determine if episode is done
        # Each task = 1 step; easy → medium → hard → done
        progression = ["easy", "medium", "hard"]
        current_idx = progression.index(task_key)
        done = current_idx == 2  # done after hard task

        if not done:
            next_key = progression[current_idx + 1]
            self._current_task_key = next_key
            self._current_bug = random.choice(TASKS[next_key]["bugs"])
            self._state.tasks_completed.append(task_key)

        return TriageObservation(
            bug_report=self._current_bug,
            task_id=self._current_task_key,
            score=self._state.total_score / max(self._state.step_count, 1),
            feedback=feedback,
            done=done,
            reward=score,
        )

    @property
    def state(self) -> TriageState:
        return self._state