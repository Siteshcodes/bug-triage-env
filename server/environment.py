# server/environment.py
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

import uuid
from openenv.core.env_server.interfaces import Environment
from model import TriageAction, TriageObservation, TriageState, BugReport
from task import grade_action, sample_bug


TASK_PROGRESSION = ["easy", "medium", "hard"]


class BugTriageEnvironment(Environment):
    """
    Bug Report Triage RL environment.

    Episode structure:
      Step 1 → easy task   (priority only)
      Step 2 → medium task (priority + labels + team)
      Step 3 → hard task   (priority + labels + team + milestone)

    NOTE: OpenEnv HTTP server creates a new environment instance per
    request. So __init__ auto-initializes a bug so step() always works.
    """

    def __init__(self):
        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            current_task="easy",
            step_count=0,
            total_score=0.0,
            tasks_completed=[],
        )
        self._current_task_key: str = "easy"
        self._episode_done: bool = False
        # Auto-init bug so step() works on stateless HTTP server
        self._current_bug: BugReport = sample_bug("easy")

    def reset(self) -> TriageObservation:
        """Start a fresh episode."""
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

    def step(self, action: TriageAction) -> TriageObservation:
        """Process the agent's triage action."""
        if self._episode_done:
            return TriageObservation(
                bug_report=self._current_bug,
                task_id=self._current_task_key,
                score=self._state.total_score / max(self._state.step_count, 1),
                feedback="Episode already complete. Call reset() to start a new episode.",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        task_key = self._current_task_key

        score, feedback = grade_action(task_key, self._current_bug, action)
        self._state.total_score += score
        self._state.tasks_completed.append(task_key)

        current_idx = TASK_PROGRESSION.index(task_key)
        done = current_idx == len(TASK_PROGRESSION) - 1

        if done:
            self._episode_done = True
            next_bug = self._current_bug
            next_task = self._current_task_key
        else:
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

    @property
    def state(self) -> TriageState:
        return self._state

    def get_state(self) -> TriageState:
        return self._state