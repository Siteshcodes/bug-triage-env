# server/app.py
import sys
import os
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from openenv.core.env_server import create_app
from model import TriageAction, TriageObservation
from environment import BugTriageEnvironment

app = create_app(
    BugTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="bug-triage-env",
)