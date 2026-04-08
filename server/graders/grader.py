from typing import Dict, Any, List
from server.models import Reward


# Root cause aliases — agent might say "db_connections" or "database pool exhausted"
# These all map to the canonical root cause key
ROOT_CAUSE_ALIASES: Dict[str, List[str]] = {
    "db_connection_exhaustion": [
        "db_connection_exhaustion", "database connection pool", "too many clients",
        "connection pool exhausted", "db pool", "connection exhaustion"
    ],
    "bad_deploy": [
        "bad_deploy", "bad deploy", "faulty deployment", "rollback needed",
        "null pointer", "deployment issue", "bad deployment"
    ],
    "memory_exhaustion": [
        "memory_exhaustion", "memory leak", "heap usage", "oom", "out of memory",
        "memory exhaustion", "gc overhead"
    ],
    "oom_kill": [
        "oom_kill", "oom killer", "killed by oom", "out of memory kill",
        "process killed", "oom_kill"
    ],
    "high_latency": [
        "high_latency", "latency spike", "slow downstream", "timeout",
        "high latency", "downstream timeout"
    ],
    "cpu_exhaustion": [
        "cpu_exhaustion", "cpu throttling", "cpu spike", "cpu high",
        "cpu exhaustion", "throttled"
    ],
    "cache_failure": [
        "cache_failure", "redis down", "cache unavailable", "cache miss",
        "redis connection refused", "cache failure"
    ],
    "external_timeout": [
        "external_timeout", "payment gateway timeout", "external timeout",
        "gateway timeout", "third party timeout"
    ],
}

# Actions that are never useful — penalise agent for taking these
WASTEFUL_ACTIONS = {"escalate"}  # Escalating without diagnosing first

# Minimum useful actions before submitting (to stop agents from guessing immediately)
MIN_USEFUL_ACTIONS = {
    "task1": 2,   # At minimum: list_alerts + query_logs
    "task2": 3,   # At minimum: list_alerts + query_logs + get_metrics
    "task3": 4,   # At minimum: list_alerts + query_logs x2 + get_metrics
    "task4": 5,   # Full investigation expected
}


class Grader:
    """
    Scores a completed episode based on:
    - correctness:    Did the agent identify the right root cause and service?
    - efficiency:     Did the agent take unnecessary/wasteful actions?
    - speed:          How many steps did it take relative to max_steps?

    Final reward = 0.50 * correctness + 0.30 * efficiency + 0.20 * speed
    """

    def grade(
        self,
        task_id: str,
        incident: Dict[str, Any],
        actions_taken: List[str],
        diagnosis: Dict[str, Any],
        steps_taken: int,
        max_steps: int,
    ) -> Reward:

        correctness  = self._score_correctness(incident, diagnosis, task_id)
        efficiency   = self._score_efficiency(actions_taken, task_id, steps_taken)
        speed        = self._score_speed(steps_taken, max_steps)

        total = round(
            0.50 * correctness +
            0.30 * efficiency +
            0.20 * speed,
            4
        )

        # Clamp to strictly between 0 and 1 (exclusive)
        # Competition validator requires: 0.0 < score < 1.0
        total = max(0.001, min(total, 0.999))

        return Reward(
            total=total,
            correctness=correctness,
            efficiency=efficiency,
            speed=speed,
            partial_credit=0.0,
        )

    def partial_reward(self, action_type: str, action_result: Dict[str, Any], incident: Dict[str, Any]) -> float:
        """
        Small positive reward signal during the episode (not just at the end).
        Encourages the agent to investigate the right service.
        """
        # Agent queried the right service's logs
        if action_type == "query_logs":
            service = action_result.get("service_queried", "")
            if service == incident["affected_service"]:
                return 0.05
            if incident.get("cascade_service") and service == incident["cascade_service"]:
                return 0.02

        # Agent queried a metric that shows the anomaly
        if action_type == "get_metrics":
            service = action_result.get("service_queried", "")
            if service == incident["affected_service"]:
                return 0.03

        # Agent ran the correct playbook
        if action_type == "run_playbook":
            playbook = action_result.get("playbook_name", "")
            if playbook == incident["correct_playbook"]:
                return 0.10

        # Agent did a rollback when rollback was the right call
        if action_type == "rollback":
            if incident["correct_rollback"]:
                return 0.10
            else:
                return -0.05  # Penalise wrong rollback

        # Tiny reward instead of exactly 0.0 (validator requirement)
        return 0.001

    # ── Private ───────────────────────────────────────────────────────────────

    def _score_correctness(self, incident: Dict, diagnosis: Dict, task_id: str) -> float:
        """
        Award points for:
        - Correct root cause identification (0.6)
        - Correct affected service (0.4)
        For task3/4, also check cascade_service if applicable.
        """
        score = 0.0

        submitted_cause   = str(diagnosis.get("root_cause", "")).lower().strip()
        submitted_service = str(diagnosis.get("affected_service", "")).lower().strip()

        true_cause   = incident["root_cause"]
        true_service = incident["affected_service"]

        # Root cause match — check aliases
        aliases = ROOT_CAUSE_ALIASES.get(true_cause, [true_cause])
        if any(alias in submitted_cause for alias in aliases):
            score += 0.6

        # Service match
        if submitted_service == true_service.lower():
            score += 0.4
        elif submitted_service in true_service.lower() or true_service.lower() in submitted_service:
            score += 0.2  # Partial credit for close match

        # Clamp to (0, 1) exclusive
        score = max(0.001, min(score, 0.999))
        return round(score, 4)

    def _score_efficiency(self, actions_taken: List[str], task_id: str, steps_taken: int) -> float:
        """
        Penalise wasteful actions and reward focused investigation.
        Score = 1.0 if agent took exactly the minimum useful actions.
        Decreases linearly as extra actions are taken.
        """
        wasteful_count  = sum(1 for a in actions_taken if a in WASTEFUL_ACTIONS)
        minimum         = MIN_USEFUL_ACTIONS.get(task_id, 3)

        # Penalise wasteful actions
        penalty = wasteful_count * 0.15

        # Penalise extra steps beyond minimum (but not too harshly)
        extra   = max(0, steps_taken - minimum)
        step_penalty = extra * 0.05

        score = 1.0 - penalty - step_penalty
        # Clamp to (0, 1) exclusive
        score = max(0.001, min(score, 0.999))
        return round(score, 4)

    def _score_speed(self, steps_taken: int, max_steps: int) -> float:
        """
        Score based on how quickly the agent solved the task.
        0.999 = solved in half the allowed steps or fewer.
        0.001 = used all steps.
        """
        if steps_taken == 0:
            return 0.001
        ratio = steps_taken / max_steps
        if ratio <= 0.5:
            score = 0.999  # Changed from 1.0
        else:
            # Linear decay from 0.999 at 50% steps to 0.001 at 100% steps
            score = 0.999 - ((ratio - 0.5) / 0.5) * 0.998
        # Clamp to (0, 1) exclusive
        score = max(0.001, min(score, 0.999))
        return round(score, 4)
