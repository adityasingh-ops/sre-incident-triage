---
title: SRE Incident Triage Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
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

An OpenEnv-compliant benchmark environment for training and evaluating AI agents on production incident response. This environment simulates the work of an on-call Site Reliability Engineer investigating service outages in a microservices architecture.

## Overview

When production systems fail, engineers need to quickly diagnose the problem by reading logs, checking metrics, correlating alerts across services, and identifying root causes under time pressure. This environment models that scenario with procedurally generated incidents that require systematic investigation.

The environment provides a simulated production topology with six microservices, four databases, and a realistic monitoring stack. Agents interact through a structured API to query logs, retrieve metrics, run remediation playbooks, and submit diagnoses. Each episode is fully reproducible using a random seed.

## System Architecture

The simulated production environment consists of the following services:

```
api-gateway → auth-service → user-db
           → order-service → payments-service → payment-db
                          → inventory-service → inventory-db
                                              → fraud-service
```

Incidents are generated to affect one or more services with various failure modes including database connection exhaustion, memory leaks, CPU saturation, bad deployments, cache failures, and external timeouts.

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

The environment provides four tasks with increasing difficulty:

**Task 1: Single Service Failure (Easy)**
- Maximum steps: 6
- A single service has a clear failure mode with obvious symptoms
- Logs and alerts directly indicate the root cause
- Minimal investigation required

**Task 2: Cascading Failure (Medium)**
- Maximum steps: 8
- A failure in one service causes downstream degradation
- Agent must trace the dependency chain to find the true root cause
- Requires correlating information across multiple services

**Task 3: Silent Failure (Hard)**
- Maximum steps: 10
- No critical alerts firing initially
- Business metrics show degradation
- Agent must correlate logs, metrics, and recent deployment history
- Root cause is not immediately obvious

**Task 4: Multi-Service Incident (Expert)**
- Maximum steps: 12
- Multiple services showing anomalies simultaneously
- Agent must investigate all affected services
- Identify the single underlying root cause
- Execute correct remediation
- Provide complete incident summary

## Reward Function

Episodes are scored using a weighted combination of three factors:

**Correctness (50%)**
- Root cause identification: 60% of correctness score
- Affected service identification: 40% of correctness score

**Efficiency (30%)**
- Penalty for unnecessary actions
- Penalty for redundant queries
- Bonus for focused investigation

**Speed (20%)**
- Based on steps taken relative to maximum allowed
- Faster resolution receives higher score

**Partial Rewards**
- Small positive rewards during the episode for productive actions
- Querying logs from the affected service
- Retrieving relevant metrics
- Running appropriate remediation

Final reward is clamped to the range [0.0, 1.0].

## Episode Generation

Each episode is procedurally generated from a random seed. The same seed will always produce:
- The same affected service
- The same root cause
- The same log patterns
- The same metric anomalies
- The same alert configuration

Different seeds produce different incidents, preventing memorization and requiring genuine diagnostic reasoning.

## API Reference

**POST /reset**
- Request body: `{"task_id": "task1", "seed": 42}`
- Returns: Initial observation and episode metadata
- Starts a new episode with the specified task and optional seed

**POST /step**
- Request body: `{"action": {"action_type": "query_logs", "params": {"service": "auth-service", "minutes_ago": 10}}}`
- Returns: Observation, reward, done flag, and info dict
- Executes one action in the current episode

**GET /state**
- Returns: Complete episode state including history and internal variables
- Useful for debugging and analysis

**GET /tasks**
- Returns: List of all available tasks with descriptions and parameters

**GET /health**
- Returns: Service health status

**GET /metadata**
- Returns: Environment metadata and configuration

**GET /schema**
- Returns: JSON schemas for Action, Observation, and State types

**GET /docs**
- Returns: Interactive Swagger UI for API exploration

## Installation and Usage

**Local Setup**

```bash
# Clone repository
git clone https://huggingface.co/spaces/adityasingh-op/sre-incident-triage
cd sre-incident-triage

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn server.main:app --host 0.0.0.0 --port 8000

# In another terminal, run tests
pytest tests/test_env.py -v
```

**Docker**

```bash
# Build image
docker build -t sre-triage .

# Run container
docker run -p 7860:7860 sre-triage

# Test endpoint
curl http://localhost:7860/health
```

**Running Inference**

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-huggingface-token"

# Run inference on a specific task
export TASK_ID="task1"
python inference.py

# Or run all tasks
python inference.py
```

## Baseline Performance

The following scores were obtained using the baseline inference script with different models:

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Average |
|-------|--------|--------|--------|--------|---------|
| Qwen2.5-72B-Instruct | 0.71 | 0.58 | 0.42 | 0.31 | 0.51 |
| GPT-4o | 0.84 | 0.67 | 0.55 | 0.44 | 0.63 |
| Random agent | 0.24 | 0.18 | 0.14 | 0.11 | 0.17 |

These scores demonstrate that the tasks have appropriate difficulty scaling and can differentiate between random behavior and model-based reasoning.

## Validation

The environment passes OpenEnv validation in Docker mode:

```bash
pip install openenv-core
openenv validate .
```

All required endpoints are implemented and conform to the OpenEnv specification.

## Development

**Running Tests**

```bash
# Start the server
uvicorn server.main:app --port 8000 &

# Run test suite
pytest tests/test_env.py -v

# All 25 tests should pass
```

**Project Structure**

```
.
├── server/
│   ├── main.py              # FastAPI application and endpoints
│   ├── engine.py            # Episode management and game logic
│   ├── models.py            # Pydantic models for types
│   ├── incident_generator.py # Procedural incident generation
│   └── graders/
│       └── grader.py        # Reward calculation and scoring
├── tests/
│   └── test_env.py          # Integration tests
├── inference.py             # Baseline agent implementation
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # Container configuration
└── README.md                # This file
```

## License

MIT

## Citation

If you use this environment in your research, please cite:

```
@misc{sre-incident-triage-2024,
  title={SRE Incident Triage: An OpenEnv Benchmark for Production Incident Response},
  author={Singh, Aditya},
  year={2024},
  url={https://huggingface.co/spaces/adityasingh-op/sre-incident-triage}
}
```
