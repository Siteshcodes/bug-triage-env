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

# 🐛 Bug Triage Environment v2.0

> **OpenEnv RL environment for the Meta PyTorch Hackathon x Scaler School of Technology**

A multi-step reinforcement learning environment where an AI agent investigates and triages GitHub-style bug reports — deciding priority, labels, team ownership, and milestone — just like a senior engineer would.

**Live:** [https://siteshcodes-bug-triage-env.hf.space](https://siteshcodes-bug-triage-env.hf.space)
**GitHub:** [https://github.com/Siteshcodes/bug-triage-env](https://github.com/Siteshcodes/bug-triage-env)

---

## What Makes This Different

| Feature | v1.0 (before) | v2.0 (now) |
|---------|---------------|------------|
| Episode length | 1 step (quiz) | Multi-step investigation |
| Bug pool | 15 hardcrafted | 200+ procedurally generated |
| Label matching | Exact string | Semantic (synonym-aware) |
| Concurrency | Broken (global state) | Session-based, thread-safe |
| Information reveal | Everything at once | Progressive (title → body → comments → logs) |
| Tests | None | 50+ unit & integration tests |
| Grading depth | String matching | Weighted scoring + reasoning bonus |

---

## Multi-Step Investigation

Unlike simple Q&A environments, the agent must **investigate before deciding**:

```
reset()     → Agent sees: bug title + body preview
step(read_body)      → Full description revealed
step(read_comments)  → User comments revealed
step(check_logs)     → Stack traces + severity signals revealed
step(submit, ...)    → Final triage graded (reward returned)
```

Each investigation step costs a step (out of a limited budget). The agent must learn **when it has enough information to decide correctly** — balancing accuracy vs. efficiency.

---

## Action Space

| Field | Type | Values |
|-------|------|--------|
| `action_type` | string | `read_body` · `read_comments` · `check_logs` · `check_similar` · `submit` |
| `priority` | string | `P0` · `P1` · `P2` · `P3` (only for submit) |
| `labels` | list[str] | `bug` · `performance` · `security` · `ux` · `data-integrity` · `payments` … |
| `assigned_team` | string | `backend` · `frontend` · `infra` · `security` · `devx` |
| `milestone` | string | `hotfix` · `v2.1` · `backlog` |
| `reasoning` | string | Free-form explanation (earns bonus points) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `bug_report` | BugReport | Title, body, author, labels_hint, comments, stack_trace |
| `task_id` | string | Current difficulty: `easy` / `medium` / `hard` |
| `score` | float | Score from grader (0.0–1.0) |
| `reward` | float | Reward from last action (0.0–1.0) |
| `feedback` | string | Human-readable grader feedback |
| `done` | bool | Episode complete flag |
| `body_visible` | bool | Whether full body has been revealed |
| `comments_visible` | bool | Whether comments have been revealed |
| `logs_visible` | bool | Whether logs/stack traces have been revealed |
| `steps_taken` | int | Steps used so far |
| `max_steps` | int | Maximum steps allowed |

---

## Tasks

### Task 1 — Easy: Priority Assignment
Assign a single P0–P3 priority. Up to 4 steps.
- **Grader:** `server.task:priority_match`
- **Scoring:** exact → 0.95, ±1 → 0.50, ±2 → 0.20, else → 0.05
- **Reward range:** (0.0, 1.0)

### Task 2 — Medium: Priority + Labels + Team
Assign priority, category labels, and team routing. Up to 5 steps.
- **Grader:** `server.task:priority_label_team`
- **Scoring:** priority 45% + label Jaccard (semantic) 40% + team 15%
- **Reward range:** (0.0, 1.0)

### Task 3 — Hard: Full Triage
Full triage with security escalation penalty. Up to 6 steps.
- **Grader:** `server.task:full_triage`
- **Scoring:** priority 35% + labels 30% + team 20% + milestone 15%
- **Penalty:** −0.15 for missing security escalation
- **Bonus:** up to +0.15 for relevant reasoning
- **Reward range:** (0.0, 1.0)

---

## Reward Function

- **Priority:** Graduated partial credit (0.95 → 0.50 → 0.20 → 0.05)
- **Labels:** Semantic Jaccard similarity with synonym matching (e.g., "defect" ≈ "bug")
- **Team routing:** Binary accuracy, weighted per difficulty
- **Security escalation:** Hard penalty (−0.15) for ignoring security signals
- **Reasoning bonus:** Up to +0.15 for mentioning relevant signals
- **Efficiency:** +0.05 bonus for correct answers with minimal investigation
- **Clamping:** All scores strictly within (0.0, 1.0)

---

## Procedural Bug Generation

The environment generates bugs from **7 template categories**:

| Category | Example Bugs |
|----------|-------------|
| `crash` | Service crashes, unhandled exceptions, segfaults |
| `security` | SQL injection, XSS, auth bypass, data exposure |
| `performance` | Memory leaks, slow queries, CPU spikes |
| `ui_bug` | Layout breaks, dark mode issues, accessibility |
| `data_corruption` | Race conditions, encoding issues, stale cache |
| `documentation` | Typos, outdated docs, missing guides |
| `api_bug` | Rate limiting bugs, pagination issues, webhook failures |

Each category has 5-6 title templates × 2 body templates × 6-12 variables = hundreds of unique combinations. The 15 original handcrafted bugs are preserved as a high-quality subset (40% chance per sample).

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

### Run Tests
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Run Inference (Hackathon Submission)
```bash
pip install openai openenv-core requests pydantic
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=your_hf_token_here
export ENV_BASE_URL=https://siteshcodes-bug-triage-env.hf.space
python inference.py
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API endpoint | Yes |
| `MODEL_NAME` | Model identifier for inference | Yes |
| `HF_TOKEN` | Hugging Face / API key | Yes |
| `ENV_BASE_URL` | Bug Triage environment URL | Optional |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive demo frontend |
| GET | `/health` | Health check + active sessions |
| POST | `/reset` | Start new episode (returns session_id) |
| POST | `/step` | Investigation or submit action |
| GET | `/state` | Current episode state |
| GET | `/tasks` | List all 3 tasks |
| GET | `/tasks/{id}` | Task metadata |
| GET | `/leaderboard` | Top agent scores |
| POST | `/leaderboard/submit` | Submit agent scores |

### Example: Multi-Step Episode

```bash
# 1. Reset — get a bug and session_id
curl -X POST https://siteshcodes-bug-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard"}'

# 2. Investigate — read full body (use session_id from step 1)
curl -X POST https://siteshcodes-bug-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action": {"action_type": "read_body"}}'

# 3. Investigate — read comments
curl -X POST https://siteshcodes-bug-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action": {"action_type": "read_comments"}}'

# 4. Submit triage decision
curl -X POST https://siteshcodes-bug-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action": {"action_type": "submit", "priority": "P0", "labels": ["bug", "security"], "assigned_team": "security", "milestone": "hotfix", "reasoning": "SQL injection in production — critical security vulnerability"}}'
```

---

## Inference Log Format

Structured logs per OpenEnv spec (3 tasks, each with its own block):

```
[START] task=easy env=bug-triage-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=investigate:read_body reward=0.00 done=false error=null
[STEP] step=2 action=investigate:read_comments reward=0.00 done=false error=null
[STEP] step=3 action=priority=P0,team=backend,milestone=hotfix reward=0.95 done=true error=null
[END] success=true steps=3 score=0.95 rewards=0.95

[START] task=medium env=bug-triage-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=investigate:read_body reward=0.00 done=false error=null
[STEP] step=2 action=investigate:read_comments reward=0.00 done=false error=null
[STEP] step=3 action=priority=P0,team=backend,milestone=hotfix reward=0.85 done=true error=null
[END] success=true steps=3 score=0.85 rewards=0.85

[START] task=hard env=bug-triage-env model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=investigate:read_body reward=0.00 done=false error=null
[STEP] step=2 action=investigate:read_comments reward=0.00 done=false error=null
[STEP] step=3 action=priority=P0,team=security,milestone=hotfix reward=0.92 done=true error=null
[END] success=true steps=3 score=0.92 rewards=0.92
```

---

## Project Structure

```
bug-triage-env/
├── server/
│   ├── app.py             # FastAPI routes + session management
│   ├── environment.py     # Multi-step environment + SessionManager
│   ├── task.py            # 200+ bugs (procedural + handcrafted) + semantic grading
│   ├── __init__.py
│   ├── requirements.txt
│   └── static/
│       └── index.html     # Interactive demo
├── tests/
│   ├── test_grading.py    # Grading logic tests
│   ├── test_environment.py # Environment flow tests
│   └── test_api.py        # HTTP endpoint integration tests
├── model.py               # Pydantic models (TriageAction, TriageObservation, TriageState)
├── client.py              # HTTP client (single source of truth)
├── inference.py           # Multi-step OpenAI agent (hackathon submission)
├── baseline.py            # Groq baseline agent
├── openenv.yaml           # OpenEnv spec manifest
├── Dockerfile             # Docker config
├── pyproject.toml         # Package metadata + dev deps
└── README.md
```

---

## OpenEnv Spec Compliance

| Requirement | Status |
|-------------|--------|
| Typed models (Action/Observation/State) | ✅ |
| `step()` / `reset()` / `state()` API | ✅ |
| `openenv.yaml` manifest | ✅ |
| 3+ tasks with graders (easy → hard) | ✅ |
| Reward range strictly (0.0, 1.0) | ✅ |
| Multi-step episodes | ✅ |
| Baseline inference with reproducible scores | ✅ |
| Dockerfile builds | ✅ |
| Deployed on HF Spaces | ✅ |
| Structured `[START]/[STEP]/[END]` logs | ✅ |
| Session-based concurrency | ✅ |
| 50+ automated tests | ✅ |

---

*Built for the Meta PyTorch Hackathon x Scaler School of Technology — Round 1*