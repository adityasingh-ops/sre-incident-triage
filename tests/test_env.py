"""
Full integration test suite for the SRE Incident Triage Environment.
Run with: pytest tests/test_env.py -v
"""

import pytest
import httpx

BASE = "http://localhost:8000"


@pytest.fixture(autouse=True)
def fresh_episode():
    """Reset to a known state before every test."""
    httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})


# ── Health & Discovery ────────────────────────────────────────────────────────

def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_lists_tasks():
    r = httpx.get(f"{BASE}/")
    data = r.json()
    assert "tasks" in data
    assert len(data["tasks"]) >= 4


def test_tasks_endpoint():
    r = httpx.get(f"{BASE}/tasks")
    tasks = r.json()["tasks"]
    ids = [t["id"] for t in tasks]
    assert "task1" in ids
    assert "task4" in ids


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_returns_observation():
    r = httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert obs["task_id"] == "task1"
    assert obs["step"] == 0
    assert len(obs["available_actions"]) == 7
    assert "fraud-service" in obs["task_description"]  # seed 42 = fraud-service


def test_reset_same_seed_same_incident():
    r1 = httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
    r2 = httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
    assert r1.json()["observation"]["task_description"] == \
           r2.json()["observation"]["task_description"]


def test_reset_different_seeds_different_incidents():
    httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
    r1 = httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
    r2 = httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 99})
    # Different seeds should (usually) produce different services
    d1 = r1.json()["observation"]["task_description"]
    d2 = r2.json()["observation"]["task_description"]
    assert d1 != d2  # seed 42 vs 99 produce different incidents


def test_reset_invalid_task():
    r = httpx.post(f"{BASE}/reset", json={"task_id": "task99", "seed": 1})
    assert r.status_code == 400


# ── Step: action routing ──────────────────────────────────────────────────────

def test_step_list_alerts():
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "list_alerts", "params": {}}})
    assert r.status_code == 200
    data = r.json()
    assert data["observation"]["alerts"] is not None
    assert len(data["observation"]["alerts"]) >= 1
    # At least one critical alert
    severities = [a["severity"] for a in data["observation"]["alerts"]]
    assert "critical" in severities


def test_step_query_logs_correct_service():
    """Querying the affected service (fraud-service for seed 42) gives ERROR logs."""
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "query_logs",
                                    "params": {"service": "fraud-service", "minutes_ago": 15}}})
    data = r.json()
    logs = data["observation"]["logs"]
    assert logs is not None
    levels = [l["level"] for l in logs]
    assert "ERROR" in levels  # Affected service has error logs
    assert data["reward"] > 0  # Partial reward for querying right service


def test_step_query_logs_wrong_service():
    """Querying a healthy service gives only INFO logs and zero reward."""
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "query_logs",
                                    "params": {"service": "auth-service", "minutes_ago": 15}}})
    data = r.json()
    logs = data["observation"]["logs"]
    levels = [l["level"] for l in logs]
    assert "ERROR" not in levels  # Healthy service — no errors
    assert data["reward"] == 0.0  # No partial reward for wrong service


def test_step_query_logs_unknown_service():
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "query_logs",
                                    "params": {"service": "nonexistent", "minutes_ago": 10}}})
    data = r.json()
    assert data["observation"]["error"] is not None
    assert data["reward"] == 0.0


def test_step_get_metrics():
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "get_metrics",
                                    "params": {"service": "fraud-service", "metric": "cpu_percent"}}})
    data = r.json()
    metrics = data["observation"]["metrics"]
    assert metrics is not None
    assert "cpu_percent" in metrics


def test_step_run_playbook_correct():
    """Running the correct playbook (scale_horizontal for cpu_exhaustion) gives reward."""
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "run_playbook",
                                    "params": {"name": "scale_horizontal", "service": "fraud-service"}}})
    data = r.json()
    assert data["observation"]["playbook_result"] is not None
    assert data["reward"] >= 0.10  # Correct playbook bonus


def test_step_run_playbook_unknown():
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "run_playbook",
                                    "params": {"name": "nonexistent_playbook", "service": "fraud-service"}}})
    data = r.json()
    assert data["observation"]["error"] is not None


def test_step_rollback_wrong_cause():
    """Rollback when root cause is NOT bad_deploy should give negative partial reward."""
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "rollback",
                                    "params": {"service": "fraud-service", "version": "v1.0.0"}}})
    data = r.json()
    # reward can be negative partial (clamped to 0 in StepResult)
    assert data["reward"] == 0.0


# ── Submit diagnosis ──────────────────────────────────────────────────────────

def test_submit_correct_diagnosis():
    """Correct root cause + service = high correctness score."""
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "submit_diagnosis",
                                    "params": {"root_cause": "cpu_exhaustion",
                                               "affected_service": "fraud-service"}}})
    data = r.json()
    assert data["done"] is True
    assert data["reward"] >= 0.7  # High score for correct answer


def test_submit_wrong_diagnosis():
    """Wrong root cause = low correctness, partial score from efficiency/speed only."""
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "submit_diagnosis",
                                    "params": {"root_cause": "db_connection_exhaustion",
                                               "affected_service": "auth-service"}}})
    data = r.json()
    assert data["done"] is True
    assert 0.0 <= data["reward"] <= 0.5  # Partial at best


def test_reward_in_range():
    """All rewards must be in [0.0, 1.0]."""
    actions = [
        {"action_type": "list_alerts", "params": {}},
        {"action_type": "query_logs", "params": {"service": "fraud-service", "minutes_ago": 10}},
        {"action_type": "get_metrics", "params": {"service": "fraud-service", "metric": "cpu_percent"}},
        {"action_type": "submit_diagnosis", "params": {"root_cause": "cpu_exhaustion",
                                                        "affected_service": "fraud-service"}},
    ]
    httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
    for action in actions:
        r = httpx.post(f"{BASE}/step", json={"action": action})
        reward = r.json()["reward"]
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for action {action}"


# ── Episode lifecycle ─────────────────────────────────────────────────────────

def test_step_after_done_is_safe():
    """Calling step after episode ends should not crash."""
    httpx.post(f"{BASE}/step",
               json={"action": {"action_type": "submit_diagnosis",
                                "params": {"root_cause": "cpu_exhaustion",
                                           "affected_service": "fraud-service"}}})
    r = httpx.post(f"{BASE}/step",
                   json={"action": {"action_type": "list_alerts", "params": {}}})
    assert r.status_code == 200
    assert r.json()["done"] is True


def test_state_after_episode():
    httpx.post(f"{BASE}/step",
               json={"action": {"action_type": "submit_diagnosis",
                                "params": {"root_cause": "cpu_exhaustion",
                                           "affected_service": "fraud-service"}}})
    r = httpx.get(f"{BASE}/state")
    state = r.json()
    assert state["done"] is True
    assert state["diagnosis_submitted"] is True
    assert state["final_reward"]["total"] >= 0.7


def test_max_steps_terminates_episode():
    """Episode must terminate at or before max_steps."""
    httpx.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42})
    done = False
    for i in range(20):  # task1 max_steps=6, so this will hit the limit
        r = httpx.post(f"{BASE}/step",
                       json={"action": {"action_type": "list_alerts", "params": {}}})
        if r.json()["done"]:
            done = True
            assert i < 15  # Should terminate well before 20
            break
    assert done, "Episode never terminated"


# ── All tasks reset correctly ─────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", ["task1", "task2", "task3", "task4"])
def test_all_tasks_reset(task_id):
    r = httpx.post(f"{BASE}/reset", json={"task_id": task_id, "seed": 1})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert obs["task_id"] == task_id
    assert obs["step"] == 0
    assert len(obs["task_description"]) > 20
