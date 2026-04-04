# server/environment.py
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

import uuid
import random
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from model import TriageAction, TriageObservation, TriageState, BugReport
from task import TASKS, grade_action, sample_bug


TASK_PROGRESSION = ["easy", "medium", "hard"]


class BugTriageEnvironment(Environment):
    """
    Bug Report Triage RL environment.

    Episode structure:
      Step 1 → easy task   (priority only)
      Step 2 → medium task (priority + labels + team)
      Step 3 → hard task   (priority + labels + team + milestone)

    Each reset() picks a fresh random bug from each task pool,
    so the agent never sees the same sequence twice.
    """

    def __init__(self):
        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            current_task="easy",
            step_count=0,
            total_score=0.0,
            tasks_completed=[],
    )
        self._current_bug: Optional[BugReport] = None
        self._current_task_key: str = "easy"
        self._episode_done: bool = False

    # ─────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────

    def reset(self) -> TriageObservation:
        """Start a fresh episode. Picks a random bug from the easy pool."""
        self._state = TriageState(
           episode_id=str(uuid.uuid4()),
           current_task="easy",
           step_count=0,
           total_score=0.0,
           tasks_completed=[],
    )
        self._current_task_key = "easy"
        self._episode_done = False
        self._current_bug = sample_bug("easy")

        return TriageObservation(
            bug_report=self._current_bug,
            task_id="easy",
            score=0.0,
            feedback="New episode started. Triage this bug report.",
            done=False,
            reward=0.0,
        )

    # ─────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Process the agent's triage action and return the next observation.

        - Grades the current task
        - Advances to next task (easy → medium → hard)
        - Returns done=True after the hard task is graded
        """
        # Guard: prevent stepping after episode is over
        if self._episode_done:
            assert self._current_bug is not None
            return TriageObservation(
                bug_report=self._current_bug,
                task_id=self._current_task_key,
                score=self._state.total_score / max(self._state.step_count, 1),
                feedback="Episode already complete. Call reset() to start a new episode.",
                done=True,
                reward=0.0,
            )

        # Guard: step() must be called after reset()
        assert self._current_bug is not None, "step() called before reset()"

        self._state.step_count += 1
        task_key = self._current_task_key

        # Grade the action for this task
        score, feedback = grade_action(task_key, self._current_bug, action)
        self._state.total_score += score
        self._state.tasks_completed.append(task_key)

        # Determine progression
        current_idx = TASK_PROGRESSION.index(task_key)
        done = current_idx == len(TASK_PROGRESSION) - 1  # True after hard task

        if done:
            # Episode complete — keep current bug/task for final observation
            self._episode_done = True
            next_bug = self._current_bug
            next_task = self._current_task_key
        else:
            # Advance to next task with a fresh random bug
            next_task = TASK_PROGRESSION[current_idx + 1]
            next_bug = sample_bug(next_task)
            self._current_task_key = next_task
            self._current_bug = next_bug

        avg_score = self._state.total_score / self._state.step_count

        return TriageObservation(
            bug_report=next_bug,
            task_id=next_task,
            score=round(avg_score, 3),
            feedback=feedback,
            done=done,
            reward=round(score, 3),
        )

    # ─────────────────────────────────────────
    # state() — both property and method forms
    # ─────────────────────────────────────────

    @property
    def state(self) -> TriageState:
        """Property form — used internally."""
        return self._state

    def get_state(self) -> TriageState:
        """Method form — satisfies OpenEnv spec's state() interface."""
        return self._state