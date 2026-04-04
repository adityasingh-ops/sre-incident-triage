import random
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


# Load the service topology once at import time
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "services.json")
with open(_DATA_PATH) as f:
    SERVICE_DATA = json.load(f)


class IncidentGenerator:
    """
    Generates a self-consistent fake production incident given a seed.
    Same seed always produces same incident — critical for reproducibility.

    The generated incident contains everything the engine needs:
    - which service failed and why (the ground truth, hidden from agent)
    - realistic logs the agent can query
    - metric timeseries the agent can pull
    - alerts that are currently firing
    - what actions correctly remediate the incident
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.now = datetime(2024, 3, 15, 14, 30, 0)  # Fixed "now" for reproducibility

    def generate(self, task_id: str) -> Dict[str, Any]:
        """Main entry point. Returns a full incident dict."""

        services    = SERVICE_DATA["services"]
        templates   = SERVICE_DATA["failure_templates"]
        playbooks   = SERVICE_DATA["playbooks"]

        # Pick which service failed based on task constraints
        service = self._pick_service(task_id, services)
        
        # Pick a failure mode available for this service
        failure_mode = self._pick_failure_mode(task_id, service, templates)
        template = templates[failure_mode]

        # For task2 (cascading), pick a secondary affected service
        cascade_service = None
        if task_id == "task2":
            cascade_service = self._pick_cascade_service(service, services)

        # For task3 (silent failure), pick a recent deploy as the trigger
        deploy_version = None
        if task_id in ("task3", "task4"):
            deploy_version = f"v{self.rng.randint(2,5)}.{self.rng.randint(0,9)}.{self.rng.randint(1,20)}"

        incident = {
            # Ground truth — grader uses these, agent never sees them directly
            "root_cause": template["root_cause"],
            "affected_service": service["name"],
            "cascade_service": cascade_service["name"] if cascade_service else None,
            "correct_playbook": template["remediation_playbook"],
            "correct_rollback": template["correct_rollback"],
            "deploy_version": deploy_version,
            "failure_mode": failure_mode,

            # Generated data the agent CAN access via actions
            "logs":    self._generate_logs(service, template, cascade_service, deploy_version),
            "metrics": self._generate_metrics(service, template, cascade_service),
            "alerts":  self._generate_alerts(service, template, cascade_service),

            # Task description shown to agent at episode start
            "description": self._generate_description(task_id, service, cascade_service),
        }

        return incident

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pick_service(self, task_id: str, services: List) -> Dict:
        """
        Task1: simple services with one dependency (auth, inventory, fraud).
        Task2+: complex services with multiple dependencies (order, payments, api-gateway).
        """
        if task_id == "task1":
            candidates = [s for s in services if len(s["dependencies"]) <= 1]
        else:
            candidates = [s for s in services if len(s["dependencies"]) > 1]
        return self.rng.choice(candidates)

    def _pick_failure_mode(self, task_id: str, service: Dict, templates: Dict) -> str:
        """Pick a failure mode that exists for this service."""
        available = [m for m in service["failure_modes"] if m in templates]
        # Task3/4 always use bad_deploy or cache_failure for the "silent" pattern
        if task_id in ("task3", "task4"):
            silent = [m for m in available if m in ("bad_deploy", "cache_failure", "high_latency")]
            if silent:
                return self.rng.choice(silent)
        return self.rng.choice(available)

    def _pick_cascade_service(self, primary: Dict, services: List) -> Optional[Dict]:
        """For cascading failures, pick a service that depends on the primary."""
        dependents = [s for s in services if primary["name"] in s["dependencies"]]
        if dependents:
            return self.rng.choice(dependents)
        # Fallback: pick something that shares a dependency
        return self.rng.choice([s for s in services if s["name"] != primary["name"]])

    def _generate_logs(self, service: Dict, template: Dict,
                       cascade: Optional[Dict], deploy_version: Optional[str]) -> Dict[str, List[Dict]]:
        """
        Generate log entries per service. The failing service has error logs
        matching the failure template. Healthy services have only info logs.
        """
        logs = {}
        pattern = template["log_pattern"]

        # Generate logs for every service so agent has to query the right one
        for svc in SERVICE_DATA["services"]:
            entries = []
            is_affected = svc["name"] == service["name"]
            is_cascade  = cascade and svc["name"] == cascade["name"]

            for i in range(20):
                t = self.now - timedelta(minutes=20 - i)
                ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")

                if is_affected and i >= 10:
                    # Error logs start appearing 10 min ago
                    entries.append({
                        "timestamp": ts,
                        "level": "ERROR",
                        "service": svc["name"],
                        "message": pattern,
                        "trace_id": f"trace-{self.rng.randint(100000, 999999)}"
                    })
                    if i % 3 == 0:
                        entries.append({
                            "timestamp": ts,
                            "level": "WARN",
                            "service": svc["name"],
                            "message": f"Retry attempt {i-9}/3 for upstream call",
                            "trace_id": f"trace-{self.rng.randint(100000, 999999)}"
                        })
                elif is_cascade and i >= 14:
                    entries.append({
                        "timestamp": ts,
                        "level": "WARN",
                        "service": svc["name"],
                        "message": f"Upstream {service['name']} responding slowly, timeout in 2000ms",
                        "trace_id": f"trace-{self.rng.randint(100000, 999999)}"
                    })
                else:
                    entries.append({
                        "timestamp": ts,
                        "level": "INFO",
                        "service": svc["name"],
                        "message": f"Request processed successfully in {self.rng.randint(10,120)}ms",
                        "trace_id": f"trace-{self.rng.randint(100000, 999999)}"
                    })

                # Sprinkle in a deploy log for task3/4
                if deploy_version and svc["name"] == service["name"] and i == 8:
                    entries.append({
                        "timestamp": (self.now - timedelta(minutes=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "level": "INFO",
                        "service": svc["name"],
                        "message": f"Deploy {deploy_version} rolled out, 3/3 pods healthy",
                        "trace_id": "deploy-event"
                    })

            logs[svc["name"]] = sorted(entries, key=lambda x: x["timestamp"])

        return logs

    def _generate_metrics(self, service: Dict, template: Dict, cascade: Optional[Dict]) -> Dict[str, Dict]:
        """
        Generate metric timeseries per service. The affected service shows
        anomalous values matching the failure signature.
        """
        metrics = {}
        sig = template["metric_signature"]

        value_map = {
            "normal":         lambda base: base + self.rng.uniform(-5, 5),
            "high":           lambda base: base * self.rng.uniform(2.5, 3.5),
            "very_high":      lambda base: base * self.rng.uniform(5.0, 8.0),
            "dropping":       lambda base: base * self.rng.uniform(0.1, 0.4),
            "zero":           lambda base: 0.0,
            "spike_then_drop":lambda base: base * self.rng.uniform(8, 12) if self.rng.random() > 0.5 else 0.0,
            "medium":         lambda base: base * self.rng.uniform(1.5, 2.5),
        }

        baselines = {
            "request_rate":        200,
            "error_rate":          0.5,
            "latency_p99":         150,
            "cpu_percent":         35,
            "memory_mb":           512,
            "orders_per_minute":   45,
            "transactions_per_minute": 30,
            "cache_hit_rate":      92,
            "timeout_rate":        0.1,
            "token_validation_ms": 12,
            "model_inference_ms":  80,
            "queue_depth":         10,
        }

        for svc in SERVICE_DATA["services"]:
            svc_metrics = {}
            is_affected = svc["name"] == service["name"]

            for metric_name in svc["metrics"]:
                base = baselines.get(metric_name, 100)
                datapoints = []

                for i in range(20):
                    t = self.now - timedelta(minutes=20 - i)
                    ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")

                    if is_affected and i >= 10 and metric_name in sig:
                        fn = value_map.get(sig[metric_name], value_map["normal"])
                        val = round(fn(base), 2)
                    else:
                        val = round(base + self.rng.uniform(-base*0.05, base*0.05), 2)

                    datapoints.append({"timestamp": ts, "value": max(0.0, val)})

                svc_metrics[metric_name] = {
                    "datapoints": datapoints,
                    "unit": self._metric_unit(metric_name),
                    "current": datapoints[-1]["value"],
                    "baseline": base
                }

            metrics[svc["name"]] = svc_metrics

        return metrics

    def _generate_alerts(self, service: Dict, template: Dict, cascade: Optional[Dict]) -> List[Dict]:
        """Generate firing alerts. Primary service gets a critical alert."""
        alerts = []
        fired_at = (self.now - timedelta(minutes=9)).strftime("%Y-%m-%dT%H:%M:%SZ")

        alerts.append({
            "id": f"alert-{self.rng.randint(10000, 99999)}",
            "severity": "critical",
            "name": template["alert_name"],
            "service": service["name"],
            "message": f"{template['alert_name']} on {service['name']} — see runbook SR-{self.rng.randint(100,999)}",
            "fired_at": fired_at,
            "status": "firing"
        })

        if cascade:
            alerts.append({
                "id": f"alert-{self.rng.randint(10000, 99999)}",
                "severity": "warning",
                "name": "LatencyP99Elevated",
                "service": cascade["name"],
                "message": f"Latency degraded on {cascade['name']}, possibly downstream issue",
                "fired_at": (self.now - timedelta(minutes=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "firing"
            })

        # Add a red herring alert on a healthy service to make task3/4 harder
        healthy = [s for s in SERVICE_DATA["services"]
                   if s["name"] != service["name"] and (not cascade or s["name"] != cascade["name"])]
        if healthy:
            decoy = self.rng.choice(healthy)
            alerts.append({
                "id": f"alert-{self.rng.randint(10000, 99999)}",
                "severity": "info",
                "name": "DiskUsageElevated",
                "service": decoy["name"],
                "message": f"Disk usage above 70% on {decoy['name']}, non-critical",
                "fired_at": (self.now - timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "firing"
            })

        return alerts

    def _generate_description(self, task_id: str, service: Dict, cascade: Optional[Dict]) -> str:
        descriptions = {
            "task1": (
                f"You are the on-call SRE. An alert just fired for {service['name']}. "
                f"Investigate the logs and metrics to identify the root cause and submit your diagnosis."
            ),
            "task2": (
                f"Multiple alerts are firing. {service['name']} appears degraded and "
                f"{cascade['name'] if cascade else 'a downstream service'} is also showing issues. "
                f"Determine which service is the true root cause and what is causing the cascade."
            ),
            "task3": (
                f"Business metrics show orders-per-minute has dropped 40% in the last 15 minutes. "
                f"No obvious critical alerts. Dig through logs, metrics, and recent deploys to find the silent failure."
            ),
            "task4": (
                f"A complex multi-service degradation is underway. Three services are showing anomalies. "
                f"Identify the root cause, run the correct remediation, and submit a complete incident report "
                f"including affected services, root cause, and recommended long-term fix."
            ),
        }
        return descriptions.get(task_id, descriptions["task1"])

    def _metric_unit(self, metric_name: str) -> str:
        units = {
            "request_rate": "req/s", "error_rate": "%", "latency_p99": "ms",
            "cpu_percent": "%", "memory_mb": "MB", "orders_per_minute": "orders/min",
            "transactions_per_minute": "txn/min", "cache_hit_rate": "%",
            "timeout_rate": "%", "token_validation_ms": "ms",
            "model_inference_ms": "ms", "queue_depth": "messages"
        }
        return units.get(metric_name, "")
