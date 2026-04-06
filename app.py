"""
SRE Incident Triage - Interactive Gradio UI
Production-grade interface for the OpenEnv SRE benchmark
"""
import os
import json
import time
import requests
import subprocess
import sys
from typing import Optional, Dict, List, Tuple
import gradio as gr
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = "http://localhost:8000"  # Backend runs on 8000

# ── Styling ──────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.incident-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.metric-box {
    background: #f8f9fa;
    padding: 15px;
    border-left: 4px solid #667eea;
    margin: 10px 0;
    border-radius: 5px;
}
.alert-critical {
    background: #fee;
    border-left: 4px solid #dc3545;
    padding: 10px;
    margin: 5px 0;
    border-radius: 4px;
}
.alert-warning {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 10px;
    margin: 5px 0;
    border-radius: 4px;
}
.log-entry {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.service-topology {
    background: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.score-display {
    font-size: 48px;
    font-weight: bold;
    color: #667eea;
    text-align: center;
    padding: 20px;
}
"""

# ── API Helpers ──────────────────────────────────────────────────────────────
def api_reset(task_id: str, seed: Optional[int] = None) -> Dict:
    """Reset environment and start new episode"""
    try:
        payload = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        response = requests.post(f"{ENV_BASE_URL}/reset", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_step(action_type: str, params: Dict) -> Dict:
    """Take a step in the environment"""
    try:
        action = {"action_type": action_type, "params": params}
        response = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_get_tasks() -> Dict:
    """Get available tasks"""
    try:
        response = requests.get(f"{ENV_BASE_URL}/tasks", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_get_leaderboard() -> List[Dict]:
    """Get leaderboard"""
    try:
        response = requests.get(f"{ENV_BASE_URL}/leaderboard", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return []


# ── State Management ─────────────────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.current_obs = None
        self.episode_history = []
        self.total_reward = 0.0
        self.done = False
        self.seed = None
        self.task_id = None

state = SessionState()


# ── Formatters ───────────────────────────────────────────────────────────────
def format_service_topology() -> str:
    """Generate ASCII service topology diagram"""
    return """
```
┌──────────────┐
│ api-gateway  │
└──────┬───────┘
       │
       ├─────► auth-service ──────► user-db
       │
       └─────► order-service ──┬──► inventory-service ──┬──► inventory-db
                                │                        │
                                │                        └──► fraud-service
                                │
                                └──► payments-service ──────► payment-db
```
"""


def format_alerts(alerts: List[Dict]) -> str:
    """Format alerts with severity colors"""
    if not alerts:
        return "✅ No alerts firing"
    
    html = ""
    for alert in alerts:
        severity = alert.get("severity", "warning")
        css_class = "alert-critical" if severity == "critical" else "alert-warning"
        html += f'<div class="{css_class}">'
        html += f'<b>{alert.get("name", "Unknown Alert")}</b><br>'
        html += f'Service: {alert.get("service", "N/A")} | Severity: {severity.upper()}<br>'
        html += f'{alert.get("description", "")}'
        html += '</div>'
    return html


def format_logs(logs: List[Dict]) -> str:
    """Format logs in terminal style"""
    if not logs:
        return "No logs available"
    
    html = '<div class="log-entry">'
    for log in logs[-10:]:  # Show last 10 logs
        timestamp = log.get("timestamp", "")
        level = log.get("level", "INFO")
        message = log.get("message", "")
        service = log.get("service", "")
        
        level_color = {
            "ERROR": "#f85149",
            "WARN": "#d29922",
            "INFO": "#58a6ff"
        }.get(level, "#d4d4d4")
        
        html += f'<span style="color: #8b949e">{timestamp}</span> '
        html += f'<span style="color: {level_color}">[{level}]</span> '
        html += f'<span style="color: #79c0ff">{service}</span> '
        html += f'<span style="color: #d4d4d4">{message}</span><br>'
    html += '</div>'
    return html


def format_metrics(metrics: Dict) -> str:
    """Format metrics as cards"""
    if not metrics:
        return "No metrics available"
    
    html = ""
    for service, metric_data in metrics.items():
        if isinstance(metric_data, dict):
            for metric_name, values in metric_data.items():
                if isinstance(values, dict):
                    current = values.get("current", 0)
                    baseline = values.get("baseline", 0)
                    diff = current - baseline
                    diff_pct = (diff / baseline * 100) if baseline != 0 else 0
                    
                    status_color = "#dc3545" if abs(diff_pct) > 20 else "#28a745"
                    
                    html += '<div class="metric-box">'
                    html += f'<b>{service} - {metric_name}</b><br>'
                    html += f'Current: <span style="color: {status_color}; font-weight: bold;">{current:.2f}</span> '
                    html += f'(Baseline: {baseline:.2f})<br>'
                    html += f'Change: <span style="color: {status_color};">{diff_pct:+.1f}%</span>'
                    html += '</div>'
    return html


def format_observation(obs: Dict) -> Tuple[str, str, str, str, str]:
    """Format observation into UI components"""
    task_desc = obs.get("task_description", "No incident description")
    step = obs.get("step", 0)
    feedback = obs.get("last_action_feedback", "Waiting for action...")
    error = obs.get("error")
    
    # Incident card
    incident_html = f'''
    <div class="incident-card">
        <h2>🚨 Incident Brief - Step {step}</h2>
        <p>{task_desc}</p>
        <p><b>Last Action:</b> {feedback}</p>
        {f'<p style="color: #ffcccc;"><b>⚠️ Error:</b> {error}</p>' if error else ''}
    </div>
    '''
    
    # Alerts
    alerts_html = format_alerts(obs.get("alerts"))
    
    # Logs
    logs_html = format_logs(obs.get("logs"))
    
    # Metrics
    metrics_html = format_metrics(obs.get("metrics"))
    
    # Playbook/Rollback results
    results = ""
    if obs.get("playbook_result"):
        results += f'<div class="metric-box"><b>Playbook Result:</b> {obs["playbook_result"]}</div>'
    if obs.get("rollback_result"):
        results += f'<div class="metric-box"><b>Rollback Result:</b> {obs["rollback_result"]}</div>'
    
    return incident_html, alerts_html, logs_html, metrics_html, results


# ── Action Handlers ──────────────────────────────────────────────────────────
def start_episode(task_choice: str, seed_input: str) -> Tuple:
    """Initialize a new episode"""
    global state
    state = SessionState()
    
    task_map = {
        "Task 1: Single Service Failure (Easy)": "task1",
        "Task 2: Cascading Failure (Medium)": "task2",
        "Task 3: Silent Failure (Hard)": "task3",
        "Task 4: Multi-Service Incident (Expert)": "task4",
    }
    
    task_id = task_map.get(task_choice, "task1")
    seed = int(seed_input) if seed_input.strip() else None
    
    result = api_reset(task_id, seed)
    
    if "error" in result:
        return f"❌ Error: {result['error']}", "", "", "", "", "Error starting episode", ""
    
    state.current_obs = result["observation"]
    state.seed = result["seed"]
    state.task_id = task_id
    state.episode_history.append(f"Episode started: {task_id}, seed={state.seed}")
    
    incident, alerts, logs, metrics, results = format_observation(state.current_obs)
    history = "\n".join(state.episode_history)
    
    status = f"✅ Episode started | Task: {task_id} | Seed: {state.seed} | Score: 0.00"
    
    return incident, alerts, logs, metrics, results, status, history


def execute_action(action_type: str, service: str, minutes_ago: str, metric: str, 
                   playbook: str, version: str, severity: str, message: str,
                   root_cause: str, affected_service: str) -> Tuple:
    """Execute an action and update UI"""
    global state
    
    if state.current_obs is None:
        return "⚠️ No active episode. Start one first!", "", "", "", "", "No episode", ""
    
    if state.done:
        return "Episode completed. Start a new one!", "", "", "", "", f"Final Score: {state.total_reward:.2f}", ""
    
    # Build params based on action type
    params = {}
    if action_type == "query_logs":
        params = {"service": service, "minutes_ago": int(minutes_ago) if minutes_ago else 10}
    elif action_type == "get_metrics":
        params = {"service": service, "metric": metric}
    elif action_type == "run_playbook":
        params = {"name": playbook, "service": service}
    elif action_type == "rollback":
        params = {"service": service, "version": version}
    elif action_type == "escalate":
        params = {"severity": severity, "message": message}
    elif action_type == "submit_diagnosis":
        params = {"root_cause": root_cause, "affected_service": affected_service}
    
    result = api_step(action_type, params)
    
    if "error" in result:
        return f"❌ Error: {result['error']}", "", "", "", "", "Error", ""
    
    state.current_obs = result["observation"]
    reward = result.get("reward", 0.0)
    state.total_reward += reward
    state.done = result.get("done", False)
    
    state.episode_history.append(
        f"Step {state.current_obs['step']}: {action_type} → reward={reward:.2f}"
    )
    
    incident, alerts, logs, metrics, results = format_observation(state.current_obs)
    history = "\n".join(state.episode_history[-20:])  # Last 20 steps
    
    status_emoji = "🎉" if state.done else "⚙️"
    status = f"{status_emoji} Step {state.current_obs['step']} | Score: {state.total_reward:.2f}"
    if state.done:
        status += " | ✅ COMPLETE"
    
    return incident, alerts, logs, metrics, results, status, history


def get_leaderboard_table() -> str:
    """Fetch and format leaderboard"""
    entries = api_get_leaderboard()
    
    if not entries:
        return "No leaderboard entries yet. Complete an episode to appear here!"
    
    html = """
    <table style="width:100%; border-collapse: collapse;">
        <tr style="background: #667eea; color: white;">
            <th style="padding: 10px;">Rank</th>
            <th>Model</th>
            <th>Task</th>
            <th>Score</th>
            <th>Steps</th>
            <th>Correct</th>
            <th>Timestamp</th>
        </tr>
    """
    
    for i, entry in enumerate(entries[:20], 1):
        bg = "#f8f9fa" if i % 2 == 0 else "white"
        correct_icon = "✅" if entry.get("root_cause_correct") else "❌"
        html += f"""
        <tr style="background: {bg};">
            <td style="padding: 8px; text-align: center;"><b>{i}</b></td>
            <td>{entry.get('model_name', 'Unknown')}</td>
            <td>{entry.get('task_id', 'N/A')}</td>
            <td><b>{entry.get('score', 0):.3f}</b></td>
            <td>{entry.get('steps_taken', 0)}</td>
            <td style="text-align: center;">{correct_icon}</td>
            <td style="font-size: 11px;">{entry.get('timestamp', '')[:19]}</td>
        </tr>
        """
    
    html += "</table>"
    return html


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_interface():
    html_content = open("sre_triage.html", "r", encoding="utf-8").read()

    with gr.Blocks(title="SRE Incident Triage") as demo:
        gr.HTML(html_content)

    return demo

# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    import sys
    
    print("🚀 Starting SRE Incident Triage Environment...")
    print("=" * 60)
    
    # Start FastAPI backend as a separate process
    print("📡 Starting FastAPI backend on port 8000...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", 
         "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give backend time to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(5)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("✅ Backend is ready!")
    except:
        print("⚠️  Backend may not be ready yet, but continuing...")
    
    print("🎮 Starting Gradio UI on port 7860...")
    print("=" * 60)
    print("🌐 Open your browser to: http://localhost:7860")
    print("=" * 60)
    
    try:
        demo = build_interface()
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            share=False
        )
    finally:
        # Cleanup: terminate backend when Gradio exits
        print("\n🛑 Shutting down backend...")
        backend_process.terminate()
        backend_process.wait()
