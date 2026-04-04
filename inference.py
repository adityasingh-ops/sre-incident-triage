"""
Baseline inference script — SRE Incident Triage Environment
Mandatory stdout format: [START] [STEP] [END]
Runs all 4 tasks sequentially, seeds 42/43/44/45 for reproducibility.
"""

import os
import json
import textwrap
import requests
from typing import List, Optional
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "sre-incident-triage"
TEMPERATURE  = 0.2
MAX_TOKENS   = 400

TASK_SEEDS = {
    "task1": 42,
    "task2": 43,
    "task3": 44,
    "task4": 45,
}

# ── Mandatory log format ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior Site Reliability Engineer (SRE) triaging a production incident.
Investigate methodically, then submit your diagnosis.

RESPOND WITH ONLY A SINGLE JSON OBJECT. No explanation, no markdown, no extra text.

Available actions:
  {"action_type": "list_alerts",        "params": {}}
  {"action_type": "query_logs",         "params": {"service": "<name>", "minutes_ago": <int>}}
  {"action_type": "get_metrics",        "params": {"service": "<name>", "metric": "<name>"}}
  {"action_type": "run_playbook",       "params": {"name": "<playbook>", "service": "<name>"}}
  {"action_type": "rollback",           "params": {"service": "<name>", "version": "<ver>"}}
  {"action_type": "escalate",           "params": {"severity": "P1|P2", "message": "<msg>"}}
  {"action_type": "submit_diagnosis",   "params": {"root_cause": "<cause>", "affected_service": "<name>"}}

Investigation strategy:
1. list_alerts first — identify the critical alert and which service it's on
2. query_logs on the CRITICAL service — look for ERROR patterns
3. get_metrics on the critical service — confirm the anomaly
4. run_playbook with the correct remediation (or rollback if bad_deploy)
5. submit_diagnosis with root_cause and affected_service

Valid root causes:
  db_connection_exhaustion | bad_deploy | memory_exhaustion | oom_kill |
  high_latency | cpu_exhaustion | cache_failure | external_timeout

Services available:
  api-gateway | auth-service | order-service | payments-service | inventory-service | fraud-service

IMPORTANT: Submit your diagnosis within the step limit. Do not waste steps.
Output ONLY valid JSON. Nothing else.
""").strip()


def build_prompt(obs: dict, history: List[str]) -> str:
    parts = [
        f"Step: {obs.get('step', '?')}",
        f"Task: {obs.get('task_description', '')}",
        f"Feedback: {obs.get('last_action_feedback', '')}",
    ]
    if obs.get("error"):
        parts.append(f"ERROR: {obs['error']}")
    if obs.get("alerts"):
        parts.append(f"Alerts:\n{json.dumps(obs['alerts'], indent=2)}")
    if obs.get("logs"):
        # Show last 8 log lines — enough to find the pattern
        shown = obs["logs"][-8:]
        parts.append(f"Logs (last {len(shown)} of {len(obs['logs'])}):\n{json.dumps(shown, indent=2)}")
    if obs.get("metrics"):
        # Summarise to current values only — saves tokens
        try:
            summary = {}
            for svc, mdata in obs["metrics"].items():
                if isinstance(mdata, dict):
                    summary[svc] = {m: v.get("current") for m, v in mdata.items()}
            parts.append(f"Metrics (current):\n{json.dumps(summary, indent=2)}")
        except Exception:
            parts.append(f"Metrics:\n{json.dumps(obs['metrics'], indent=2)}")
    if obs.get("playbook_result"):
        parts.append(f"Playbook result: {obs['playbook_result']}")
    if obs.get("rollback_result"):
        parts.append(f"Rollback result: {obs['rollback_result']}")

    if history:
        parts.append("Recent steps:\n" + "\n".join(history[-3:]))

    parts.append("Your next action (JSON only):")
    return "\n\n".join(parts)


def get_action(client: OpenAI, obs: dict, history: List[str]) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action_type": "list_alerts", "params": {}}


# ── Environment helpers ──────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Single episode ───────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str, seed: int) -> float:
    max_steps   = {"task1": 6, "task2": 8, "task3": 10, "task4": 12}[task_id]
    rewards:  List[float] = []
    history:  List[str]   = []
    steps_taken = 0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = env_reset(task_id, seed)
        obs        = reset_resp["observation"]
        done       = False

        for step in range(1, max_steps + 1):
            if done:
                break

            action_dict = get_action(client, obs, history)
            action_str  = json.dumps(action_dict, separators=(",", ":"))

            try:
                result  = env_step(action_dict)
                obs     = result["observation"]
                reward  = float(result.get("reward", 0.0))
                done    = bool(result.get("done", False))
                error   = obs.get("error")
            except Exception as e:
                reward, done, error = 0.0, False, str(e)
                print(f"[DEBUG] step error: {e}", flush=True)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"step={step} action={action_dict.get('action_type')} "
                f"reward={reward:.2f} feedback={str(obs.get('last_action_feedback',''))[:60]}"
            )

            if done:
                break

        total   = sum(rewards)
        success = total >= 0.5

    except Exception as e:
        print(f"[DEBUG] episode error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, rewards=rewards)
    return sum(rewards)


# ── Main: run all 4 tasks ────────────────────────────────────────────────────

def main():
    # Check which task to run — default runs all 4
    task_override = os.getenv("TASK_ID", "all")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if task_override != "all" and task_override in TASK_SEEDS:
        tasks_to_run = {task_override: TASK_SEEDS[task_override]}
    else:
        tasks_to_run = TASK_SEEDS

    all_scores = {}
    for task_id, seed in tasks_to_run.items():
        print(f"\n{'='*50}", flush=True)
        print(f"[INFO] Starting {task_id} (seed={seed})", flush=True)
        print(f"{'='*50}", flush=True)
        score = run_episode(client, task_id, seed)
        all_scores[task_id] = round(score, 4)

    # Final summary
    print(f"\n{'='*50}", flush=True)
    print("[SUMMARY] All task scores:", flush=True)
    for tid, sc in all_scores.items():
        print(f"  {tid}: {sc:.4f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  average: {avg:.4f}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()
