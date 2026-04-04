# client.py
import os
import requests
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
    bug = BugReport(**bug_data)
    return TriageObservation(
        bug_report=bug,
        task_id=data.get("task_id", "easy"),
        score=data.get("score", 0.0),
        feedback=data.get("feedback", ""),
        done=data.get("done", False),
        reward=data.get("reward", 0.0),
    )


class BugTriageClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url
            or os.getenv("ENV_BASE_URL", "https://siteshcodes-bug-triage-env.hf.space")
        ).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self) -> TriageObservation:
        response = self.session.post(f"{self.base_url}/reset", json={}, timeout=30)
        response.raise_for_status()
        data = response.json()
        obs_data = data.get("observation", data)
        return _parse_observation(obs_data)

    def step(self, action: TriageAction) -> StepResult:
        from dataclasses import asdict
        payload = {"action": action.dict()}
        response = self.session.post(f"{self.base_url}/step", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        obs_data = data.get("observation", data)
        obs = _parse_observation(obs_data)
        return StepResult(
            observation=obs,
            reward=data.get("reward", obs.reward) or 0.0,
            done=data.get("done", obs.done),
            info={},
        )

    def state(self) -> dict:
        response = self.session.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()