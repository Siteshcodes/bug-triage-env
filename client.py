# client.py — Single source of truth for environment client
import os
import requests
from typing import Optional, List
from model import TriageAction, TriageObservation, BugReport


class StepResult:
    """Result returned by env.step()."""
    def __init__(self, observation: TriageObservation, reward: float,
                 done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


def _parse_observation(data: dict) -> TriageObservation:
    """Parse a JSON dict into a TriageObservation."""
    bug_data = data["bug_report"]
    try:
        bug = BugReport.model_validate(bug_data)
    except Exception:
        bug = BugReport(**bug_data)

    return TriageObservation(
        bug_report=bug,
        task_id=data.get("task_id", "easy"),
        score=data.get("score", 0.0),
        feedback=data.get("feedback", ""),
        done=data.get("done", False),
        reward=data.get("reward", 0.0),
        body_visible=data.get("body_visible", False),
        comments_visible=data.get("comments_visible", False),
        logs_visible=data.get("logs_visible", False),
        similar_visible=data.get("similar_visible", False),
        steps_taken=data.get("steps_taken", 0),
        max_steps=data.get("max_steps", 6),
    )


class BugTriageClient:
    """HTTP client for the Bug Triage Environment server."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url
            or os.getenv("ENV_BASE_URL", "https://siteshcodes-bug-triage-env.hf.space")
        ).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._session_id: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def reset(self, task_id: str = "easy", seed: int = None) -> TriageObservation:
        """Start a new episode. Stores session_id for subsequent step() calls."""
        payload = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if self._session_id:
            payload["session_id"] = self._session_id

        response = self.session.post(
            f"{self.base_url}/reset", json=payload, timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        self._session_id = data.get("session_id")
        obs_data = data.get("observation", data)
        return _parse_observation(obs_data)

    def step(self, action: TriageAction) -> StepResult:
        """Send an action (investigation or submit) and get the result."""
        try:
            action_dict = action.model_dump()
        except AttributeError:
            action_dict = action.dict()

        payload = {"action": action_dict}
        if self._session_id:
            payload["session_id"] = self._session_id

        response = self.session.post(
            f"{self.base_url}/step", json=payload, timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        obs_data = data.get("observation", data)
        obs = _parse_observation(obs_data)

        reward = data.get("reward", obs.reward) or 0.0
        reward = float(reward)

        # Update session_id if server returned one
        if "session_id" in data:
            self._session_id = data["session_id"]

        return StepResult(
            observation=obs,
            reward=reward,
            done=data.get("done", obs.done),
            info=data.get("info", {}),
        )

    def investigate(self, action_type: str) -> StepResult:
        """Shortcut for investigation actions."""
        action = TriageAction(action_type=action_type)
        return self.step(action)

    def submit(self, priority: str, labels: List[str] = None,
               assigned_team: str = "backend", milestone: str = "backlog",
               reasoning: str = "") -> StepResult:
        """Shortcut for submitting the final triage decision."""
        action = TriageAction(
            action_type="submit",
            priority=priority,
            labels=labels or ["bug"],
            assigned_team=assigned_team,
            milestone=milestone,
            reasoning=reasoning,
        )
        return self.step(action)

    def state(self) -> dict:
        """Get current environment state."""
        params = {}
        if self._session_id:
            params["session_id"] = self._session_id
        response = self.session.get(
            f"{self.base_url}/state", params=params, timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()