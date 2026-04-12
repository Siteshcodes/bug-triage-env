# server/environment.py
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")
import uuid
import time
from typing import Dict, Optional, Tuple
from openenv.core.env_server.interfaces import Environment
from model import TriageAction, TriageObservation, TriageState, BugReport
from task import grade_action, sample_bug

VALID_TASKS = ["easy", "medium", "hard"]

MAX_STEPS_PER_TASK = {"easy": 4, "medium": 5, "hard": 6}

TASKS_META = [
    {"id": "easy", "name": "Priority Assignment",
     "grader": "server.task:priority_match",
     "difficulty": "easy", "reward_range": [0.0, 1.0],
     "description": "Investigate a bug report and assign a P0-P3 priority. "
                    "Use investigation actions to gather info before submitting."},
    {"id": "medium", "name": "Priority Labels and Team",
     "grader": "server.task:priority_label_team",
     "difficulty": "medium", "reward_range": [0.0, 1.0],
     "description": "Investigate and assign priority, labels, and team routing. "
                    "More investigation steps available."},
    {"id": "hard", "name": "Full Triage",
     "grader": "server.task:full_triage",
     "difficulty": "hard", "reward_range": [0.0, 1.0],
     "description": "Full triage with priority, labels, team, milestone, "
                    "and security escalation penalty. Investigation is critical."},
]

INVESTIGATION_ACTIONS = {"read_body", "read_comments", "check_logs", "check_similar"}


class BugTriageEnvironment(Environment):
    """Multi-step bug triage environment with progressive information reveal."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._current_task_key: str = "easy"
        self._episode_done: bool = False
        self._current_bug: Optional[BugReport] = None
        self._current_answer: Optional[dict] = None
        self._step_count: int = 0
        self._max_steps: int = 4
        self._actions_taken: list = []

        # Progressive visibility
        self._body_visible: bool = False
        self._comments_visible: bool = False
        self._logs_visible: bool = False
        self._similar_visible: bool = False

        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            current_task="easy",
            step_count=0,
            total_score=0.0,
            tasks_completed=[],
            actions_taken=[],
        )

    def get_metadata(self):
        try:
            from openenv.core.env_server.types import EnvironmentMetadata
            return EnvironmentMetadata(
                name="bug-triage-env",
                description="Multi-step bug triage RL environment with progressive "
                            "information reveal and 3 difficulty levels",
                version="2.0.0",
                author="Siteshcodes",
                tasks=TASKS_META,
            )
        except Exception:
            return {
                "name": "bug-triage-env",
                "description": "Multi-step bug triage RL environment",
                "version": "2.0.0",
                "author": "Siteshcodes",
                "tasks": TASKS_META,
            }

    def _build_observation(self, score=0.0, feedback="", done=False,
                           reward=0.0) -> TriageObservation:
        """Build observation with current visibility state."""
        bug = self._current_bug

        # Create a visibility-filtered view of the bug
        visible_bug = BugReport(
            id=bug.id,
            title=bug.title,
            body=bug.body if self._body_visible else bug.body[:120] + "..." if len(bug.body) > 120 else bug.body,
            author=bug.author,
            labels_hint=bug.labels_hint,
            comments=bug.comments if self._comments_visible else [],
            severity_signals=bug.severity_signals if self._logs_visible else [],
            related_bugs=bug.related_bugs if self._similar_visible else [],
            stack_trace=bug.stack_trace if self._logs_visible else "",
            affected_component=bug.affected_component if self._logs_visible else "",
        )

        return TriageObservation(
            bug_report=visible_bug,
            task_id=self._current_task_key,
            score=round(score, 3),
            feedback=feedback,
            done=done,
            reward=round(reward, 3),
            body_visible=self._body_visible,
            comments_visible=self._comments_visible,
            logs_visible=self._logs_visible,
            similar_visible=self._similar_visible,
            steps_taken=self._step_count,
            max_steps=self._max_steps,
        )

    def reset(self, task_id: str = "easy", seed: int = None,
              episode_id: str = None, **kwargs) -> TriageObservation:
        """Start a fresh episode for the given task."""
        if task_id not in VALID_TASKS:
            task_id = "easy"

        self._current_task_key = task_id
        self._episode_done = False
        self._step_count = 0
        self._max_steps = MAX_STEPS_PER_TASK.get(task_id, 4)
        self._actions_taken = []

        # Reset visibility — title + truncated body are always visible
        self._body_visible = False
        self._comments_visible = False
        self._logs_visible = False
        self._similar_visible = False

        # Sample a bug and its answer
        self._current_bug, self._current_answer = sample_bug(task_id, seed=seed)

        self._state = TriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            current_task=task_id,
            step_count=0,
            total_score=0.0,
            tasks_completed=[],
            actions_taken=[],
        )

        feedback = (
            f"Episode started for task: {task_id}. "
            f"You see the bug title and a preview. "
            f"Use investigation actions (read_body, read_comments, check_logs, check_similar) "
            f"to reveal more information, then submit your triage. "
            f"You have {self._max_steps} steps max."
        )

        return self._build_observation(
            score=0.0, feedback=feedback, done=False, reward=0.0,
        )

    def step(self, action: TriageAction) -> TriageObservation:
        """Process agent's action — either investigate or submit final triage."""
        if self._episode_done:
            return self._build_observation(
                score=0.0,
                feedback="Episode already complete. Call reset() to start a new episode.",
                done=True, reward=0.0,
            )

        self._step_count += 1
        self._state.step_count = self._step_count
        action_type = getattr(action, "action_type", "submit")
        self._actions_taken.append(action_type)
        self._state.actions_taken = list(self._actions_taken)

        # Check if max steps reached — force submission
        if self._step_count >= self._max_steps and action_type != "submit":
            action_type = "submit"

        # --- Investigation actions ---
        if action_type in INVESTIGATION_ACTIONS:
            feedback = self._handle_investigation(action_type)
            return self._build_observation(
                score=0.0, feedback=feedback, done=False, reward=0.0,
            )

        # --- Submit action ---
        return self._handle_submission(action)

    def _handle_investigation(self, action_type: str) -> str:
        """Reveal information based on the investigation action."""
        if action_type == "read_body":
            if self._body_visible:
                return "Full body already revealed. Choose another action or submit."
            self._body_visible = True
            return (
                f"Full bug description revealed. "
                f"Steps used: {self._step_count}/{self._max_steps}."
            )

        elif action_type == "read_comments":
            if self._comments_visible:
                return "Comments already revealed. Choose another action or submit."
            self._comments_visible = True
            n = len(self._current_bug.comments)
            return (
                f"Revealed {n} comment(s). "
                f"Steps used: {self._step_count}/{self._max_steps}."
            )

        elif action_type == "check_logs":
            if self._logs_visible:
                return "Logs already revealed. Choose another action or submit."
            self._logs_visible = True
            has_trace = bool(self._current_bug.stack_trace)
            return (
                f"System logs revealed. {'Stack trace available.' if has_trace else 'No stack trace.'} "
                f"Steps used: {self._step_count}/{self._max_steps}."
            )

        elif action_type == "check_similar":
            if self._similar_visible:
                return "Similar bugs already revealed. Choose another action or submit."
            self._similar_visible = True
            n = len(self._current_bug.related_bugs)
            return (
                f"Found {n} related bug(s). "
                f"Steps used: {self._step_count}/{self._max_steps}."
            )

        return f"Unknown investigation action: {action_type}"

    def _handle_submission(self, action: TriageAction) -> TriageObservation:
        """Grade the agent's final triage submission."""
        score, feedback = grade_action(
            self._current_task_key, self._current_bug, action,
            answer=self._current_answer,
        )

        # Apply time efficiency bonus/penalty
        # Fewer steps = better (if the answer is good)
        investigation_steps = self._step_count - 1  # subtract the submit step
        if investigation_steps == 0 and score >= 0.7:
            # Got it right without investigating — impressive!
            efficiency_bonus = 0.05
            feedback += " | ⚡ Efficiency bonus: +0.05 (correct with minimal investigation)"
        elif investigation_steps >= 3 and score >= 0.7:
            # Took many steps but got it right — slight penalty for slowness
            efficiency_penalty = 0.02 * (investigation_steps - 2)
            score = score - efficiency_penalty
            feedback += f" | ⏱ Time penalty: -{efficiency_penalty:.2f} ({investigation_steps} investigation steps)"
        elif investigation_steps == 0 and score < 0.5:
            # Rushed and got it wrong — penalty
            feedback += " | ⚠ Consider investigating before submitting next time"

        if investigation_steps == 0 and score >= 0.7:
            score += 0.05

        score = max(0.01, min(0.99, score))

        self._state.total_score += score
        self._state.tasks_completed.append(self._current_task_key)
        self._episode_done = True

        return self._build_observation(
            score=score, feedback=feedback, done=True, reward=score,
        )

    @property
    def state(self) -> TriageState:
        return self._state

    def get_state(self) -> TriageState:
        return self._state


# ---------------------------------------------------------------------------
#  SESSION MANAGER — handles concurrent sessions safely
# ---------------------------------------------------------------------------

class SessionManager:
    """Thread-safe session management for multiple concurrent agents."""

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 600):
        self._sessions: Dict[str, BugTriageEnvironment] = {}
        self._timestamps: Dict[str, float] = {}
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds

    def create_session(self) -> Tuple[str, BugTriageEnvironment]:
        """Create a new session and return (session_id, env)."""
        self._cleanup_expired()
        session_id = str(uuid.uuid4())
        env = BugTriageEnvironment()
        self._sessions[session_id] = env
        self._timestamps[session_id] = time.time()
        # Enforce max after adding
        while len(self._sessions) > self._max_sessions:
            oldest = min(self._timestamps, key=self._timestamps.get)
            if oldest == session_id:
                break
            self._sessions.pop(oldest, None)
            self._timestamps.pop(oldest, None)
        return session_id, env

    def get_session(self, session_id: str) -> Optional[BugTriageEnvironment]:
        """Get an existing session's environment, or None if expired/missing."""
        if session_id not in self._sessions:
            return None
        # Refresh TTL on access
        self._timestamps[session_id] = time.time()
        return self._sessions[session_id]

    def remove_session(self, session_id: str) -> None:
        """Remove a session after episode completes."""
        self._sessions.pop(session_id, None)
        self._timestamps.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        """Remove sessions that exceeded TTL."""
        now = time.time()
        expired = [
            sid for sid, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._timestamps.pop(sid, None)

        # Also enforce max sessions (remove oldest)
        while len(self._sessions) > self._max_sessions:
            oldest = min(self._timestamps, key=self._timestamps.get)
            self._sessions.pop(oldest, None)
            self._timestamps.pop(oldest, None)

    @property
    def active_count(self) -> int:
        return len(self._sessions)