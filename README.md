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

# 🐛 Bug Triage Environment

> **OpenEnv RL environment for the Meta PyTorch Hackathon x Scaler School of Technology**

An OpenEnv reinforcement learning environment where an AI agent triages GitHub-style bug reports — assigning priority, labels, team ownership, and milestone — exactly as a senior engineer would.

**Live:** [https://siteshcodes-bug-triage-env.hf.space](https://siteshcodes-bug-triage-env.hf.space)
**GitHub:** [https://github.com/Siteshcodes/bug-triage-env](https://github.com/Siteshcodes/bug-triage-env)

---

## Why This Environment?

Every software team triages dozens of bug reports weekly. Getting prioritization wrong delays critical fixes and wastes engineering time. This environment trains and evaluates agents on real triage decision-making, with graders that reflect actual engineering judgment.

**Key features:**
- 🎯 Simulates a real-world engineering task (not a game or toy)
- 📊 3 tasks of increasing difficulty with deterministic graders
- 🔄 Meaningful partial-credit reward function
- 🛡️ Security escalation penalty for missed critical vulnerabilities
- 📦 Full OpenEnv spec compliance: `step()` / `reset()` / `state()`

---

## Action Space

| Field           | Type      | Values                                          |
|-----------------|-----------|-------------------------------------------------|
| `priority`      | string    | `P0` · `P1` · `P2` · `P3`                      |
| `labels`        | list[str] | `bug` · `performance` · `security` · `ux` · `data-integrity` · `payments` … |
| `assigned_team` | string    | `backend` · `frontend` · `infra` · `security` · `devx` |
| `milestone`     | string    | `hotfix` · `v2.1` · `backlog`                   |
| `reasoning`     | string    | Free-form explanation of triage decision         |

## Observation Space

| Field        | Type      | Description                              |
|--------------|-----------|------------------------------------------|
| `bug_report` | BugReport | Title, body, author, labels_hint, comments |
| `task_id`    | string    | Current difficulty: `easy` / `medium` / `hard` |
| `score`      | float     | Score from grader (0.0–1.0)              |
| `reward`     | float     | Reward from last action (0.0–1.0)        |
| `feedback`   | string    | Human-readable grader feedback           |
| `done`       | bool      | Episode complete flag                    |

---

## Tasks

### Task 1 — Easy: Priority Assignment
Assign a single P0–P3 priority to a bug report.
- **Grader:** `server.task:priority_match`
- **Scoring:** exact match → 0.95, one level off → 0.50, else → 0.05
- **Weight:** priority 100%
- **Reward range:** (0.0, 1.0) — strictly exclusive

### Task 2 — Medium: Priority + Labels + Team
Assign priority, category labels, and team routing.
- **Grader:** `server.task:priority_label_team`
- **Scoring:** priority 45% + label Jaccard similarity 40% + team routing 15%
- **Reward range:** (0.0, 1.0) — strictly exclusive

### Task 3 — Hard: Full Triage
Full triage: priority, labels, team, and milestone. Security escalation failures are penalized.
- **Grader:** `server.task:full_triage`
- **Scoring:** priority 35% + labels 30% + team 20% + milestone 15%
- **Penalty:** −0.15 for missing security escalation (e.g., SQL injection assigned to `backend` instead of `security`)
- **Reward range:** (0.0, 1.0) — strictly exclusive

---

## Reward Function

Rewards provide meaningful partial-credit signals at every step:
- **Priority:** Close-but-wrong gets partial credit (0.50 for 1-level off vs 0.05 for 2+ levels off vs 0.95 for exact match)
- **Labels:** Jaccard similarity between predicted and expected label sets (continuous signal)
- **Team routing:** Binary accuracy, weighted per task difficulty
- **Security escalation:** Hard penalty (−0.15) discourages ignoring critical security signals
- **Clamping:** All scores strictly within (0.0, 1.0) — never exactly 0 or 1

---

## Setup

### Run Locally
```bash
git clone https://github.com/Siteshcodes/bug-triage-env.git
cd bug-triage-env
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

### Run Inference (Hackathon Submission Script)
```bash
pip install openai openenv-core requests pydantic
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=your_hf_token_here
export ENV_BASE_URL=https://siteshcodes-bug-triage-env.hf.space
python inference.py
```

### Environment Variables

| Variable       | Description                          | Required |
|----------------|--------------------------------------|----------|
| `API_BASE_URL` | LLM API endpoint                     | Yes      |
| `MODEL_NAME`   | Model identifier for inference       | Yes      |
| `HF_TOKEN`     | Hugging Face / API key               | Yes      |
| `ENV_BASE_URL` | Bug Triage environment URL           | Optional |

---

## Baseline Scores

Evaluated with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router (temperature=0):

| Task       | Difficulty | Score |
|------------|------------|-------|
| Easy       | easy       | 0.95  |
| Medium     | medium     | 0.50  |
| Hard       | hard       | 0.85  |
| **Average**|            | **0.77** |

> Scores vary per run due to random bug sampling from a pool of 5 bugs per task.

---

## API Endpoints

| Method | Endpoint         | Description                        |
|--------|------------------|------------------------------------|
| GET    | `/`              | Health check                       |
| POST   | `/reset`         | Start new episode for a task       |
| POST   | `/step`          | Submit triage action               |
| GET    | `/state`         | Get current episode state          |
| GET    | `/tasks`         | List all tasks with grader info    |
| GET    | `/tasks/{id}`    | Get specific task metadata         |

### Example: Reset + Step

```bash
# Reset for easy task
curl -X POST https://siteshcodes-bug-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Submit triage action
curl -X POST https://siteshcodes-bug-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"priority": "P0", "labels": ["bug"], "assigned_team": "backend", "milestone": "hotfix", "reasoning": "App crash affecting all users"}}'
```

---

## Inference Log Format

The inference script emits structured logs per the OpenEnv spec:

```
[START] task=easy env=bug-triage-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=priority=P0,team=backend,milestone=hotfix reward=0.95 done=true error=null
[END] success=true steps=1 score=0.95 rewards=0.95

[START] task=medium env=bug-triage-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=priority=P0,team=backend,milestone=hotfix reward=0.85 done=true error=null
[END] success=true steps=1 score=0.85 rewards=0.85

[START] task=hard env=bug-triage-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=priority=P0,team=security,milestone=hotfix reward=0.72 done=true error=null
[END] success=true steps=1 score=0.72 rewards=0.72
```

Each task gets its own `[START]` → `[STEP]` → `[END]` block.

---

## Project Structure

```
bug-triage-env/
├── server/
│   ├── app.py             # FastAPI + OpenEnv stateful endpoints
│   ├── environment.py     # BugTriageEnvironment (reset/step/state)
│   ├── task.py            # 15 bug reports + 3 graders
│   ├── __init__.py
│   └── requirements.txt
├── model.py               # Pydantic models (TriageAction, TriageObservation, TriageState)
├── inference.py           # OpenAI client submission script (per-task logs)
├── openenv.yaml           # OpenEnv spec manifest (3 tasks with graders)
├── Dockerfile             # Docker container config
├── pyproject.toml         # Package metadata
└── README.md
```

---

## OpenEnv Spec Compliance

| Requirement                         | Status |
|-------------------------------------|--------|
| Typed models (Action/Observation/State) | ✅ |
| `step()` / `reset()` / `state()` API   | ✅ |
| `openenv.yaml` manifest                | ✅ |
| 3+ tasks with graders (easy→hard)      | ✅ |
| Reward range strictly (0.0, 1.0)       | ✅ |
| Baseline inference with reproducible scores | ✅ |
| Dockerfile builds                       | ✅ |
| Deployed on HF Spaces                  | ✅ |
| Structured `[START]/[STEP]/[END]` logs  | ✅ |

---

*Built for the Meta PyTorch Hackathon x Scaler School of Technology — Round 1*