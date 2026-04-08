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

An OpenEnv reinforcement learning environment where an AI agent triages GitHub-style bug reports — assigning priority, labels, team ownership, and milestone — exactly as a senior engineer would.

## Why this environment?

Every software team triages dozens of bug reports weekly. Getting prioritization wrong delays critical fixes and wastes engineering time. This environment trains and evaluates agents on real triage decision-making, with graders that reflect actual engineering judgment.

## Action space

| Field           | Type      | Values                                          |
|-----------------|-----------|-------------------------------------------------|
| `priority`      | string    | `P0` `P1` `P2` `P3`                             |
| `labels`        | list[str] | `bug` `performance` `security` `ux` `docs` …   |
| `assigned_team` | string    | `backend` `frontend` `infra` `security` `devx` |
| `milestone`     | string    | `hotfix` `v2.1` `backlog`                       |
| `reasoning`     | string    | Free-form explanation                           |

## Observation space

| Field        | Type      | Description                              |
|--------------|-----------|------------------------------------------|
| `bug_report` | BugReport | Title, body, author, comments            |
| `task_id`    | string    | Current difficulty: easy / medium / hard |
| `score`      | float     | Cumulative score this episode            |
| `reward`     | float     | Reward from last action (0.0–1.0)        |
| `feedback`   | string    | Human-readable grader feedback           |
| `done`       | bool      | Episode complete flag                    |

## Tasks

### Task 1 — Easy (Priority labeling)
Agent assigns a single P0–P3 priority to a bug report.
- Grader: exact match = 1.0, one level off = 0.5, else 0.0
- Grader weight: priority 100%

### Task 2 — Medium (Priority + labels + team)
Agent assigns priority, category labels, and team routing.
- Grader: priority 45% + label Jaccard similarity 40% + team routing 15%

### Task 3 — Hard (Full triage)
Agent must assign priority, labels, team, and milestone. Security escalation failures are penalized.
- Grader: priority 35% + labels 30% + team 20% + milestone 15%
- Penalty: −0.15 for missing security escalation

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

### Run inference (hackathon submission script)
```bash
pip install openai openenv-core
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=your_hf_token_here
export ENV_BASE_URL=https://siteshcodes-bug-triage-env.hf.space
python inference.py
```

### Run baseline (development script)
```bash
pip install groq openenv-core
export GROQ_API_KEY=your_key_here
python baseline.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Baseline scores

Evaluated with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router (temperature=0):

| Task       | Score |
|------------|-------|
| Easy       | 1.000 |
| Medium     | 0.500 |
| Hard       | 1.000 |
| **Avg**    | **0.833** |

Scores vary per run due to random bug sampling from a pool of 5 bugs per task.

## Project structure

```
bug-triage-env/
├── server/
│   ├── app.py           # FastAPI + OpenEnv entrypoint
│   ├── environment.py   # BugTriageEnvironment core logic
│   ├── task.py          # Bug reports + graders
│   └── requirements.txt
├── model.py             # Dataclass models
├── client.py            # HTTP client
├── baseline.py          # Groq development script
├── inference.py         # OpenAI client submission script
├── openenv.yaml         # OpenEnv spec metadata
├── Dockerfile
└── README.md
```