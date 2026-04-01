---
title: Bug Triage Env
emoji: 🐛
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
---

# Bug Triage Environment 🐛

An OpenEnv reinforcement learning environment where an AI agent
triages GitHub-style bug reports — assigning priority, labels,
team ownership, and milestone — exactly as a senior engineer would.

## Why this environment?

Every software team triages dozens of bug reports weekly.
Getting prioritization wrong delays critical fixes and wastes
engineering time. This environment trains and evaluates agents
on real triage decision-making, with graders that reflect
actual engineering judgment.

## Action space

| Field           | Type        | Values                                      |
|-----------------|-------------|---------------------------------------------|
| `priority`      | string      | `P0` `P1` `P2` `P3`                         |
| `labels`        | list[str]   | `bug` `performance` `security` `ux` `docs`… |
| `assigned_team` | string      | `backend` `frontend` `infra` `security` `devx` |
| `milestone`     | string      | `hotfix` `v2.1` `backlog`                   |
| `reasoning`     | string      | Free-form explanation                       |

## Observation space

| Field        | Type        | Description                              |
|--------------|-------------|------------------------------------------|
| `bug_report` | BugReport   | Title, body, author, comments            |
| `task_id`    | string      | Current difficulty: easy / medium / hard |
| `score`      | float       | Cumulative score this episode            |
| `reward`     | float       | Reward from last action (0.0–1.0)        |
| `feedback`   | string      | Human-readable grader feedback           |
| `done`       | bool        | Episode complete flag                    |

## Tasks

### Task 1 — Easy (Priority labeling)
Agent assigns a single P0–P3 priority to a bug report.
- Grader: exact match = 1.0, one level off = 0.5, else 0.0
- Expected baseline score: ~0.75

### Task 2 — Medium (Priority + label classification)
Agent assigns priority AND a set of category labels.
- Grader: 50% priority score + 50% Jaccard label similarity
- Expected baseline score: ~0.60

### Task 3 — Hard (Full triage)
Agent must assign priority, labels, team, and milestone.
Security escalation failures are penalized.
- Grader: 40% priority + 35% labels + 25% team routing
- Penalty: −0.15 for missing security escalation
- Expected baseline score: ~0.45

## Reward function

Rewards are provided at every step (not just end of episode):
- Partial credit for close-but-not-exact priority (0.5 vs 0.0 vs 1.0)
- Label overlap via Jaccard similarity (continuous signal)
- Team routing accuracy (binary, but weighted)
- Security escalation penalty discourages ignoring critical signals

## Setup

### Run locally
```bash
git clone https://huggingface.co/spaces/Siteshcodes/bug-triage-env
cd bug-triage-env
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

### Run baseline
```bash
pip install groq openenv-core websockets
export GROQ_API_KEY=your_key_here
python baseline.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Baseline scores

Evaluated with `llama-3.3-70b-versatile` via Groq (temperature=0):

| Task   | Score |
|--------|-------|
| Easy   | ~0.75 |
| Medium | ~0.60 |
| Hard   | ~0.45 |
| **Avg**| **~0.60** |

## Project structure
```
bug-triage-env/
├── server/
│   ├── app.py           # FastAPI + OpenEnv entrypoint
│   ├── environment.py   # BugTriageEnvironment core logic
│   ├── task.py          # Bug reports + graders
│   └── requirements.txt
├── model.py             # Dataclass models
├── client.py            # WebSocket client
├── baseline.py          # Groq inference script
├── openenv.yaml         # OpenEnv spec metadata
├── Dockerfile
└── README.md
```