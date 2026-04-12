# tests/test_environment.py
"""Tests for the environment logic in server/environment.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "server"))

import pytest
from model import TriageAction, TriageObservation
from server.environment import BugTriageEnvironment, SessionManager


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = BugTriageEnvironment()
        obs = env.reset(task_id="easy")
        assert isinstance(obs, TriageObservation)
        assert obs.bug_report is not None
        assert obs.done is False
        assert obs.task_id == "easy"

    def test_reset_different_tasks(self):
        env = BugTriageEnvironment()
        for task_id in ["easy", "medium", "hard"]:
            obs = env.reset(task_id=task_id)
            assert obs.task_id == task_id
            assert obs.done is False

    def test_reset_invalid_task_defaults_to_easy(self):
        env = BugTriageEnvironment()
        obs = env.reset(task_id="nonexistent")
        assert obs.task_id == "easy"

    def test_reset_shows_truncated_body(self):
        env = BugTriageEnvironment()
        obs = env.reset(task_id="easy")
        # Body should be truncated (not fully visible) on reset
        assert obs.body_visible is False

    def test_reset_hides_comments(self):
        env = BugTriageEnvironment()
        obs = env.reset(task_id="easy")
        assert obs.comments_visible is False

    def test_reset_clears_previous_state(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        env.step(TriageAction(action_type="submit", priority="P0"))
        # Reset should clear everything
        obs = env.reset(task_id="medium")
        assert obs.done is False
        assert obs.task_id == "medium"
        assert obs.steps_taken == 0


class TestEnvironmentInvestigation:
    def test_read_body_reveals_full_body(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        obs = env.step(TriageAction(action_type="read_body"))
        assert obs.body_visible is True
        assert obs.done is False
        assert obs.steps_taken == 1

    def test_read_comments_reveals_comments(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        obs = env.step(TriageAction(action_type="read_comments"))
        assert obs.comments_visible is True
        assert obs.done is False

    def test_check_logs_reveals_logs(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        obs = env.step(TriageAction(action_type="check_logs"))
        assert obs.logs_visible is True
        assert obs.done is False

    def test_duplicate_investigation_gives_feedback(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        env.step(TriageAction(action_type="read_body"))
        obs = env.step(TriageAction(action_type="read_body"))
        assert "already" in obs.feedback.lower()

    def test_step_count_increments(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        obs1 = env.step(TriageAction(action_type="read_body"))
        assert obs1.steps_taken == 1
        obs2 = env.step(TriageAction(action_type="read_comments"))
        assert obs2.steps_taken == 2


class TestEnvironmentSubmission:
    def test_submit_returns_done(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        obs = env.step(TriageAction(action_type="submit", priority="P0"))
        assert obs.done is True

    def test_submit_returns_valid_score(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        obs = env.step(TriageAction(action_type="submit", priority="P0"))
        assert 0 < obs.score < 1
        assert 0 < obs.reward < 1

    def test_investigate_then_submit(self):
        env = BugTriageEnvironment()
        env.reset(task_id="medium")
        env.step(TriageAction(action_type="read_body"))
        env.step(TriageAction(action_type="read_comments"))
        obs = env.step(TriageAction(
            action_type="submit", priority="P0",
            labels=["bug"], assigned_team="backend",
        ))
        assert obs.done is True
        assert 0 < obs.score < 1

    def test_double_submit_stays_done(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        env.step(TriageAction(action_type="submit", priority="P0"))
        obs = env.step(TriageAction(action_type="submit", priority="P1"))
        assert obs.done is True
        assert "already complete" in obs.feedback.lower()

    def test_max_steps_forces_submit(self):
        env = BugTriageEnvironment()
        obs = env.reset(task_id="easy")
        max_steps = obs.max_steps

        # Use all steps investigating
        for _ in range(max_steps - 1):
            obs = env.step(TriageAction(action_type="read_body"))
            if obs.done:
                break

        # This should force a submit even if action_type is investigate
        if not obs.done:
            obs = env.step(TriageAction(
                action_type="read_comments",  # will be forced to submit
                priority="P0",
            ))


class TestEnvironmentState:
    def test_state_tracks_steps(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        env.step(TriageAction(action_type="read_body"))
        state = env.get_state()
        assert state.step_count == 1
        assert "read_body" in state.actions_taken

    def test_state_tracks_completed_tasks(self):
        env = BugTriageEnvironment()
        env.reset(task_id="easy")
        env.step(TriageAction(action_type="submit", priority="P0"))
        state = env.get_state()
        assert "easy" in state.tasks_completed


class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager(max_sessions=10, ttl_seconds=60)
        session_id, env = mgr.create_session()
        assert session_id is not None
        assert isinstance(env, BugTriageEnvironment)
        assert mgr.active_count == 1

    def test_get_session(self):
        mgr = SessionManager()
        session_id, env = mgr.create_session()
        retrieved = mgr.get_session(session_id)
        assert retrieved is env

    def test_get_missing_session(self):
        mgr = SessionManager()
        assert mgr.get_session("nonexistent") is None

    def test_remove_session(self):
        mgr = SessionManager()
        session_id, _ = mgr.create_session()
        mgr.remove_session(session_id)
        assert mgr.get_session(session_id) is None
        assert mgr.active_count == 0

    def test_max_sessions_enforced(self):
        mgr = SessionManager(max_sessions=3, ttl_seconds=60)
        for _ in range(5):
            mgr.create_session()
        assert mgr.active_count <= 3

    def test_multiple_sessions_independent(self):
        mgr = SessionManager()
        sid1, env1 = mgr.create_session()
        sid2, env2 = mgr.create_session()

        env1.reset(task_id="easy")
        env2.reset(task_id="hard")

        assert env1.get_state().current_task == "easy"
        assert env2.get_state().current_task == "hard"
