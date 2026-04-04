from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ─── Action Types ────────────────────────────────────────────────────────────
# These are the 7 things an agent can do, mirroring what a real SRE does
# during an incident. The agent must pick one per step.

class ActionType(str, Enum):
    query_logs     = "query_logs"       # Read logs from a service in a time window
    get_metrics    = "get_metrics"      # Pull a metric (cpu, memory, latency, error_rate)
    list_alerts    = "list_alerts"      # See what alerts are currently firing
    run_playbook   = "run_playbook"     # Execute a remediation playbook
    rollback       = "rollback"         # Roll back a service to previous deploy
    escalate       = "escalate"         # Page a human / escalate severity
    submit_diagnosis = "submit_diagnosis" # Final answer: what broke and why


# ─── Action ──────────────────────────────────────────────────────────────────
# What the agent sends to env.step(). 
# params is flexible — different action types need different params:
#   query_logs:       {"service": "api-gateway", "minutes_ago": 10}
#   get_metrics:      {"service": "payments", "metric": "error_rate"}
#   list_alerts:      {}  (no params needed)
#   run_playbook:     {"name": "restart_service", "service": "auth"}
#   rollback:         {"service": "payments", "version": "v2.1.0"}
#   escalate:         {"severity": "P1", "message": "payments is down"}
#   submit_diagnosis: {"root_cause": "db_pool_exhaustion", "affected_service": "payments"}

class Action(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)


# ─── Observation ─────────────────────────────────────────────────────────────
# What the agent sees after each step. This is the agent's "window" into
# the fake production system. Not all fields are populated every step —
# only the ones relevant to the action the agent just took.

class Observation(BaseModel):
    step: int                                      # Which step we're on
    task_id: str                                   # e.g. "task1", "task2"
    task_description: str                          # Natural language brief of the incident
    available_actions: List[str]                   # Remind agent what it can do
    
    # Populated based on action type:
    logs: Optional[List[Dict[str, Any]]] = None    # From query_logs
    metrics: Optional[Dict[str, Any]]   = None    # From get_metrics
    alerts: Optional[List[Dict[str, Any]]] = None  # From list_alerts
    playbook_result: Optional[str]      = None    # From run_playbook
    rollback_result: Optional[str]      = None    # From rollback
    escalation_result: Optional[str]   = None    # From escalate

    # Always present — gives agent a sense of progress
    last_action_feedback: Optional[str] = None    # Human-readable result of last action
    error: Optional[str]               = None    # If agent did something invalid


# ─── Reward ──────────────────────────────────────────────────────────────────
# Broken into components so we can give partial credit.
# Total reward = correctness*0.5 + efficiency*0.3 + speed*0.2
# This means an agent that finds the right answer in 3 steps beats one
# that finds it in 8 steps, which beats one that never finds it.

class Reward(BaseModel):
    total: float = 0.0             # Final 0.0–1.0 score
    correctness: float = 0.0       # Did it identify root cause correctly?
    efficiency: float = 0.0        # Did it take unnecessary actions?
    speed: float = 0.0             # How fast relative to max steps?
    partial_credit: float = 0.0    # Partial signal during episode


# ─── Step Result ─────────────────────────────────────────────────────────────
# What env.step() returns. Standard OpenEnv interface.

class StepResult(BaseModel):
    observation: Observation
    reward: float                  # Scalar reward for this step (for inference.py logging)
    done: bool                     # Is the episode over?
    info: Dict[str, Any] = Field(default_factory=dict)  # Extra metadata for graders


# ─── Episode State ───────────────────────────────────────────────────────────
# Internal state of a running episode. Stored in the engine, not exposed
# directly to the agent — but returned by env.state() for debugging.

class EpisodeState(BaseModel):
    task_id: str
    seed: int
    current_step: int = 0
    max_steps: int = 10
    done: bool = False
    incident: Dict[str, Any] = Field(default_factory=dict)  # The generated scenario
    actions_taken: List[str] = Field(default_factory=list)
    diagnosis_submitted: bool = False
    final_reward: Optional[Reward] = None


# ─── Reset Response ──────────────────────────────────────────────────────────
# What env.reset() returns — the initial observation to kick off an episode.

class ResetResponse(BaseModel):
    observation: Observation
    episode_id: str
    seed: int


# ─── Leaderboard Entry ───────────────────────────────────────────────────────
# Stored after each completed episode for the /leaderboard endpoint.

class LeaderboardEntry(BaseModel):
    model_name: str
    task_id: str
    seed: int
    score: float
    steps_taken: int
    root_cause_correct: bool
    timestamp: str
