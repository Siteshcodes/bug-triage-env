# tests/test_api.py
"""Integration tests for the FastAPI endpoints."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "server"))

import pytest

# These tests require fastapi and httpx
try:
    from fastapi.testclient import TestClient
    from server.app import app
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="FastAPI/httpx not installed")


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in ("ok", "healthy")


class TestTaskEndpoints:
    def test_list_tasks(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) == 3
        ids = [t["id"] for t in tasks]
        assert "easy" in ids
        assert "medium" in ids
        assert "hard" in ids

    def test_get_specific_task(self, client):
        r = client.get("/tasks/easy")
        assert r.status_code == 200
        assert r.json()["id"] == "easy"

    def test_get_nonexistent_task(self, client):
        r = client.get("/tasks/impossible")
        assert r.status_code == 404


class TestResetEndpoint:
    def test_reset_returns_observation(self, client):
        r = client.post("/reset", json={"task_id": "easy"})
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "session_id" in data
        assert data["done"] is False

    def test_reset_with_empty_body(self, client):
        r = client.post("/reset", json={})
        assert r.status_code == 200

    def test_reset_returns_bug_report(self, client):
        r = client.post("/reset", json={"task_id": "medium"})
        data = r.json()
        obs = data["observation"]
        assert "bug_report" in obs
        assert "title" in obs["bug_report"]


class TestStepEndpoint:
    def test_investigation_step(self, client):
        # Reset first
        r = client.post("/reset", json={"task_id": "easy"})
        session_id = r.json()["session_id"]

        # Investigate
        r = client.post("/step", json={
            "session_id": session_id,
            "action": {"action_type": "read_body"},
        })
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is False

    def test_submit_step(self, client):
        # Reset
        r = client.post("/reset", json={"task_id": "easy"})
        session_id = r.json()["session_id"]

        # Submit
        r = client.post("/step", json={
            "session_id": session_id,
            "action": {
                "action_type": "submit",
                "priority": "P0",
                "labels": ["bug"],
                "assigned_team": "backend",
            },
        })
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert 0 < data["reward"] < 1

    def test_full_episode_flow(self, client):
        # Reset
        r = client.post("/reset", json={"task_id": "hard"})
        assert r.status_code == 200
        session_id = r.json()["session_id"]

        # Investigate: read body
        r = client.post("/step", json={
            "session_id": session_id,
            "action": {"action_type": "read_body"},
        })
        assert r.status_code == 200
        assert r.json()["done"] is False

        # Investigate: read comments
        r = client.post("/step", json={
            "session_id": session_id,
            "action": {"action_type": "read_comments"},
        })
        assert r.status_code == 200
        assert r.json()["done"] is False

        # Submit triage
        r = client.post("/step", json={
            "session_id": session_id,
            "action": {
                "action_type": "submit",
                "priority": "P0",
                "labels": ["bug", "security"],
                "assigned_team": "security",
                "milestone": "hotfix",
                "reasoning": "Critical security vulnerability in production",
            },
        })
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert 0 < data["reward"] < 1

    def test_backward_compatible_no_session(self, client):
        """Old-style requests without session_id should still work."""
        r = client.post("/reset", json={"task_id": "easy"})
        assert r.status_code == 200

        r = client.post("/step", json={
            "action": {
                "priority": "P0",
                "labels": ["bug"],
            },
        })
        assert r.status_code == 200


class TestStateEndpoint:
    def test_state_returns_data(self, client):
        client.post("/reset", json={"task_id": "easy"})
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert "current_task" in data
        assert "step_count" in data


class TestLeaderboard:
    def test_get_empty_leaderboard(self, client):
        r = client.get("/leaderboard")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_submit_to_leaderboard(self, client):
        r = client.post("/leaderboard/submit", json={
            "agent_name": "test-agent",
            "model": "test-model",
            "scores": {"easy": 0.9, "medium": 0.7, "hard": 0.5},
            "avg_score": 0.7,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "submitted"
        assert "rank" in data
