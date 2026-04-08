import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from server.models import (
    Action, ActionType, Observation, StepResult,
    EpisodeState, ResetResponse, Reward
)
from server.incident_generator import IncidentGenerator
from server.graders.grader import Grader, clamp_score

# Max steps per task — harder tasks get more steps
MAX_STEPS = {"task1": 6, "task2": 8, "task3": 10, "task4": 12}

AVAILABLE_ACTIONS = [a.value for a in ActionType]


class EpisodeEngine:
    """
    Manages one active episode. Instantiated fresh for each reset() call.
    Holds the incident (ground truth) and processes agent actions step by step.
    """

    def __init__(self, task_id: str, seed: int):
        self.task_id    = task_id
        self.seed       = seed
        self.episode_id = str(uuid.uuid4())[:8]
        self.grader     = Grader()

        # Generate the incident (ground truth + all queryable data)
        generator     = IncidentGenerator(seed)
        self.incident = generator.generate(task_id)

        self.state = EpisodeState(
            task_id=task_id,
            seed=seed,
            max_steps=MAX_STEPS.get(task_id, 8),
        )

    def reset(self) -> ResetResponse:
        """Return the initial observation — agent sees task brief + available actions."""
        obs = Observation(
            step=0,
            task_id=self.task_id,
            task_description=self.incident["description"],
            available_actions=AVAILABLE_ACTIONS,
            alerts=None,
            last_action_feedback="Episode started. Investigate the incident using available actions.",
        )
        return ResetResponse(
            observation=obs,
            episode_id=self.episode_id,
            seed=self.seed,
        )

    def step(self, action: Action) -> StepResult:
        """Process one agent action. Returns new observation, reward, done flag."""

        # Guard: episode already over
        if self.state.done:
            return self._terminal_result("Episode already completed.")

        self.state.current_step += 1
        self.state.actions_taken.append(action.action_type.value)

        # Route to the right handler
        handler = {
            ActionType.query_logs:       self._handle_query_logs,
            ActionType.get_metrics:      self._handle_get_metrics,
            ActionType.list_alerts:      self._handle_list_alerts,
            ActionType.run_playbook:     self._handle_run_playbook,
            ActionType.rollback:         self._handle_rollback,
            ActionType.escalate:         self._handle_escalate,
            ActionType.submit_diagnosis: self._handle_submit_diagnosis,
        }.get(action.action_type)

        if not handler:
            obs = self._base_obs(error=f"Unknown action: {action.action_type}")
            return StepResult(observation=obs, reward=clamp_score(0.001), done=False)

        obs, partial, done = handler(action.params)

        # Check step limit
        if self.state.current_step >= self.state.max_steps and not done:
            done = True
            obs.last_action_feedback = (obs.last_action_feedback or "") + " [MAX STEPS REACHED]"
            partial -= 0.1  # Small penalty for running out of steps

        self.state.done = done
        # Clamp reward to strictly (0, 1)
        reward = clamp_score(partial)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self.state.current_step,
                "max_steps": self.state.max_steps,
                "episode_id": self.episode_id,
            }
        )

    def get_state(self) -> EpisodeState:
        return self.state

    # ── Action Handlers ───────────────────────────────────────────────────────

    def _handle_query_logs(self, params: Dict) -> tuple:
        service    = params.get("service", "")
        minutes    = int(params.get("minutes_ago", 10))
        all_logs   = self.incident["logs"]

        if service not in all_logs:
            services_available = list(all_logs.keys())
            obs = self._base_obs(error=f"Unknown service '{service}'. Available: {services_available}")
            return obs, clamp_score(0.001), False

        # Return only logs from the requested time window
        cutoff_minute = 20 - minutes
        logs = [
            entry for entry in all_logs[service]
            if self._log_index(entry) >= cutoff_minute
        ]

        feedback = f"Retrieved {len(logs)} log entries from {service} (last {minutes} min)"
        obs = self._base_obs(logs=logs, feedback=feedback)

        partial = self.grader.partial_reward("query_logs", {"service_queried": service}, self.incident)
        return obs, partial, False

    def _handle_get_metrics(self, params: Dict) -> tuple:
        service     = params.get("service", "")
        metric_name = params.get("metric", "")
        all_metrics = self.incident["metrics"]

        if service not in all_metrics:
            obs = self._base_obs(error=f"Unknown service '{service}'.")
            return obs, clamp_score(0.001), False

        svc_metrics = all_metrics[service]
        if metric_name and metric_name not in svc_metrics:
            obs = self._base_obs(
                error=f"Unknown metric '{metric_name}'. Available: {list(svc_metrics.keys())}"
            )
            return obs, clamp_score(0.001), False

        # Return one metric or all metrics for the service
        result = {metric_name: svc_metrics[metric_name]} if metric_name else svc_metrics
        feedback = f"Metrics for {service}" + (f" [{metric_name}]" if metric_name else " [all]")
        obs = self._base_obs(metrics=result, feedback=feedback)

        partial = self.grader.partial_reward("get_metrics", {"service_queried": service}, self.incident)
        return obs, partial, False

    def _handle_list_alerts(self, params: Dict) -> tuple:
        alerts   = self.incident["alerts"]
        feedback = f"{len(alerts)} alert(s) currently firing."
        obs      = self._base_obs(alerts=alerts, feedback=feedback)
        return obs, 0.01, False  # Tiny reward for checking alerts (good SRE behaviour)

    def _handle_run_playbook(self, params: Dict) -> tuple:
        name    = params.get("name", "")
        service = params.get("service", "")
        playbooks = {
            "restart_service":         "Service pods restarted. Health checks passing.",
            "scale_db_connections":    "Connection pool increased to 200. New read replica attached.",
            "drain_traffic":           "Traffic drained. Service removed from load balancer.",
            "enable_circuit_breaker":  "Circuit breaker enabled. Fallback responses active.",
            "scale_horizontal":        "3 new pods added. Deployment scaled to 6 replicas.",
            "switch_payment_provider": "Traffic rerouted to backup payment provider.",
            "restart_cache":           "Redis cluster flushed and restarted. Cache warming.",
            "rollback_deploy":         "Rolled back to previous stable image. Pods restarting.",
        }

        if name not in playbooks:
            obs = self._base_obs(error=f"Unknown playbook '{name}'. Available: {list(playbooks.keys())}")
            return obs, clamp_score(0.001), False

        result  = playbooks[name]
        obs     = self._base_obs(playbook_result=result, feedback=f"Playbook '{name}' executed: {result}")
        partial = self.grader.partial_reward("run_playbook", {"playbook_name": name}, self.incident)
        return obs, partial, False

    def _handle_rollback(self, params: Dict) -> tuple:
        service = params.get("service", "")
        version = params.get("version", "previous")
        result  = f"Rollback of {service} to {version} initiated. Pods restarting, ETA 90 seconds."
        obs     = self._base_obs(rollback_result=result, feedback=result)
        partial = self.grader.partial_reward("rollback", {}, self.incident)
        return obs, partial, False

    def _handle_escalate(self, params: Dict) -> tuple:
        severity = params.get("severity", "P2")
        message  = params.get("message", "")
        result   = f"Escalated as {severity}. On-call lead paged. Message: '{message}'"
        obs      = self._base_obs(escalation_result=result, feedback=result)
        return obs, clamp_score(0.001), False

    def _handle_submit_diagnosis(self, params: Dict) -> tuple:
        """Final action — grade the episode and close it."""
        self.state.diagnosis_submitted = True
        self.state.done = True

        reward_obj = self.grader.grade(
            task_id=self.task_id,
            incident=self.incident,
            actions_taken=self.state.actions_taken,
            diagnosis=params,
            steps_taken=self.state.current_step,
            max_steps=self.state.max_steps,
        )
        self.state.final_reward = reward_obj

        feedback = (
            f"Diagnosis submitted. "
            f"Score: {reward_obj.total:.3f} | "
            f"Correctness: {reward_obj.correctness:.2f} | "
            f"Efficiency: {reward_obj.efficiency:.2f} | "
            f"Speed: {reward_obj.speed:.2f}"
        )
        obs = self._base_obs(feedback=feedback)
        return obs, reward_obj.total, True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _base_obs(
        self,
        logs=None, metrics=None, alerts=None,
        playbook_result=None, rollback_result=None, escalation_result=None,
        feedback: str = "", error: str = None
    ) -> Observation:
        return Observation(
            step=self.state.current_step,
            task_id=self.task_id,
            task_description=self.incident["description"],
            available_actions=AVAILABLE_ACTIONS,
            logs=logs,
            metrics=metrics,
            alerts=alerts,
            playbook_result=playbook_result,
            rollback_result=rollback_result,
            escalation_result=escalation_result,
            last_action_feedback=feedback,
            error=error,
        )

    def _terminal_result(self, feedback: str) -> StepResult:
        obs = self._base_obs(feedback=feedback)
        return StepResult(observation=obs, reward=clamp_score(0.001), done=True)

    def _log_index(self, log_entry: Dict) -> int:
        """Approximate which minute slot a log belongs to (for time windowing)."""
        try:
            ts = datetime.strptime(log_entry["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
            ref = datetime(2024, 3, 15, 14, 10, 0)
            return int((ts - ref).total_seconds() / 60)
        except Exception:
            return 0
