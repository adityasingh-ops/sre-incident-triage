---
title: SRE Incident Triage Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - sre
  - agent-benchmark
  - devops
  - incident-response
---

# SRE Incident Triage Environment

> An OpenEnv-compliant benchmark for training and evaluating AI agents on **production incident response** — the most time-critical, high-stakes task in software engineering.

## 🎮 Interactive UI

This Space features a **production-grade Gradio interface** that lets you:

- 🎯 **Play incidents manually** or watch AI agents solve them
- 📊 **Real-time visualization** of alerts, logs, and metrics
- 🏆 **Live leaderboard** to track performance
- 🗺️ **Service topology** diagram showing microservice dependencies
- 📈 **Step-by-step scoring** with correctness, efficiency, and speed breakdown

**[Launch the UI above ↑]** to start triaging incidents!

## Why this exists

Every company running software at scale has on-call engineers who get paged at 2am, stare at dashboards, and have to figure out what broke and why — often under pressure, with incomplete information. This environment simulates exactly that. It is the first OpenEnv benchmark focused on **SRE reasoning and remediation**.

An agent that scores well here can:
- Read noisy logs across multiple services
- Correlate metrics to identify anomalies
- Distinguish root causes from symptoms
- Select the correct remediation action
- Do all of this efficiently, without unnecessary steps

## Environment overview

The agent is dropped into a fake production system with 6 microservices, 4 databases, and a realistic alert stack. A procedurally generated incident has occurred — seeded for full reproducibility. The agent must investigate using structured actions and submit a diagnosis before running out of steps.

**Fake production topology:**

```
api-gateway → auth-service → user-db
           → order-service → payments-service → payment-db
                           → inventory-service → inventory-db
                                               → fraud-service
```

## Action space

| Action | Parameters | Description |
|---|---|---|
| `list_alerts` | none | See all currently firing alerts |
| `query_logs` | `service`, `minutes_ago` | Read logs from a specific service |
| `get_metrics` | `service`, `metric` | Pull metric timeseries |
| `run_playbook` | `name`, `service` | Execute a remediation playbook |
| `rollback` | `service`, `version` | Roll back a bad deploy |
| `escalate` | `severity`, `message` | Page a human |
| `submit_diagnosis` | `root_cause`, `affected_service` | Final answer — ends episode |

## Observation space

Each step returns:
- `task_description` — natural language incident brief
- `logs` — list of log entries (if `query_logs` was called)
- `metrics` — metric timeseries with current value and baseline
- `alerts` — firing alerts with severity and service
- `playbook_result` / `rollback_result` — remediation feedback
- `last_action_feedback` — human-readable result of last action
- `error` — populated if action was invalid

## Tasks

| ID | Name | Difficulty | Max Steps | Description |
|---|---|---|---|---|
| `task1` | Single service failure | Easy | 6 | One service has a clear failure. Logs and alerts point directly to root cause. |
| `task2` | Cascading failure | Medium | 8 | A failure in one service causes degradation downstream. Agent must trace the cascade. |
| `task3` | Silent failure | Hard | 10 | No critical alerts. Business metrics dropping. Agent must correlate logs, metrics, and a recent deploy. |
| `task4` | Multi-service incident | Expert | 12 | Three services show anomalies. Full investigation + remediation + incident report required. |

## Reward function

```
reward = 0.50 × correctness + 0.30 × efficiency + 0.20 × speed
```

- **Correctness** — did the agent identify the right root cause (0.6) and affected service (0.4)?
- **Efficiency** — did the agent take unnecessary actions? Penalises wasteful steps.
- **Speed** — how quickly did the agent solve it relative to max steps?
- **Partial rewards** — small rewards during the episode for querying the right service, running the correct playbook, etc.

## Root causes

`db_connection_exhaustion` | `bad_deploy` | `memory_exhaustion` | `oom_kill` | `high_latency` | `cpu_exhaustion` | `cache_failure` | `external_timeout`

## Procedural generation

Every `reset()` call generates a unique incident from a seed. Same seed = same incident (reproducible). Different seed = different service, different failure mode, different logs and metrics. The agent cannot memorise answers — it must actually reason.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start episode `{"task_id": "task1", "seed": 42}` |
| POST | `/step` | Take action `{"action": {"action_type": "...", "params": {}}}` |
| GET | `/state` | Current episode state (debug) |
| GET | `/tasks` | List all tasks |
| GET | `/leaderboard` | Scores from completed episodes |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |

## Setup

### Via HuggingFace Space (Recommended)

Just click the UI above! The interactive interface lets you:
1. Select a task difficulty (Easy → Expert)
2. Choose a seed for reproducibility
3. Take actions step-by-step or watch an agent
4. View real-time alerts, logs, and metrics
5. Submit your diagnosis and see your score

### Local Development

```bash
# Clone and install
git clone https://huggingface.co/spaces/YOUR_USERNAME/sre-incident-triage
cd sre-incident-triage
pip install -r requirements.txt

# Run Gradio UI (includes FastAPI backend)
python app.py

# Or run just the API backend
uvicorn server.main:app --reload --port 8000

# Docker
docker build -t sre-env .
docker run -p 7860:7860 sre-env
```

### Run Baseline Agent (Programmatic)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export TASK_ID="task1"
python inference.py
```

## Baseline scores

| Model | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|
| Qwen2.5-72B-Instruct | 0.71 | 0.58 | 0.42 | 0.31 |
| GPT-4o | 0.84 | 0.67 | 0.55 | 0.44 |
| Random agent | 0.24 | 0.18 | 0.14 | 0.11 |

## Validate

```bash
pip install openenv-core
openenv validate
```

## 🎨 UI Features

The Gradio interface provides:

- **Interactive Playground:** Execute actions manually and see immediate feedback
- **Visual Alerts:** Color-coded critical and warning alerts
- **Terminal-style Logs:** Syntax-highlighted log viewer with timestamps
- **Metric Dashboard:** Real-time metrics with anomaly detection
- **Service Topology:** ASCII diagram of microservice dependencies
- **Episode History:** Track all actions and rewards per step
- **Leaderboard:** Compare scores across models and tasks

Perfect for:
- 🎓 **Learning** SRE incident response patterns
- 🤖 **Debugging** AI agent behavior
- 🏆 **Competing** on the leaderboard
- 📊 **Demoing** your model's capabilities

---
title: Sre Incident Triage
emoji: 📈
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
