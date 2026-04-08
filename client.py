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
    )


class BugTriageClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url
            or os.getenv("ENV_BASE_URL", "https://siteshcodes-bug-triage-env.hf.space")
        ).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self, task_id: str = "easy") -> TriageObservation:
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return _parse_observation(data.get("observation", data))

    def step(self, action: TriageAction) -> StepResult:
        try:
            action_dict = action.model_dump()   # Pydantic v2
        except AttributeError:
            action_dict = action.dict()         # Pydantic v1 fallback
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action_dict},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        obs = _parse_observation(data.get("observation", data))
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