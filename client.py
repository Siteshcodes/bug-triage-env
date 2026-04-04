# client.py
import os
import requests
from dataclasses import asdict
from typing import Optional
from model import TriageAction, TriageObservation, BugReport


class StepResult:
    def __init__(self, observation: TriageObservation, reward: float, done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


def _parse_observation(data: dict) -> TriageObservation:
    bug_data = data["bug_report"]
    bug = BugReport(
        id=bug_data["id"],
        title=bug_data["title"],
        body=bug_data["body"],
        author=bug_data["author"],
        labels_hint=bug_data.get("labels_hint", []),
        comments=bug_data.get("comments", []),
    )
    return TriageObservation(
        bug_report=bug,
        task_id=data["task_id"],
        score=data["score"],
        feedback=data["feedback"],
        done=data["done"],
        reward=data["reward"],
    )


class BugTriageClient:
    """
    HTTP REST client for BugTriageEnvironment.
    Uses POST /reset and POST /step endpoints.

    Usage:
        with BugTriageClient() as env:
            obs = env.reset()
            while not obs.done:
                action = TriageAction(...)
                result = env.step(action)
                obs = result.observation
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url
            or os.getenv("ENV_BASE_URL", "https://siteshcodes-bug-triage-env.hf.space")
        ).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self) -> TriageObservation:
        """Call POST /reset to start a new episode."""
        response = self.session.post(
            f"{self.base_url}/reset",
            json={},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return _parse_observation(data["observation"])

    def step(self, action: TriageAction) -> StepResult:
        """Call POST /step with the triage action."""
        payload = {"action": asdict(action)}
        response = self.session.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        obs = _parse_observation(data["observation"])
        return StepResult(
            observation=obs,
            reward=data.get("reward", obs.reward) or 0.0,
            done=data.get("done", obs.done),
            info={},
        )

    def state(self) -> dict:
        """Call GET /state to get current environment state."""
        response = self.session.get(
            f"{self.base_url}/state",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()