import random
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.models import (
    Action, StepResult, ResetResponse,
    EpisodeState, LeaderboardEntry
)
from server.engine import EpisodeEngine

app = FastAPI(
    title="SRE Incident Triage Environment",
    description="OpenEnv-compliant benchmark for training AI agents on production incident response.",
    version="1.0.0",
)

# ── In-memory state ───────────────────────────────────────────────────────────
# One active episode at a time (sufficient for hackathon + single-agent eval)
_active_engine: Optional[EpisodeEngine] = None
_leaderboard: List[LeaderboardEntry] = []


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1"   # task1 | task2 | task3 | task4
    seed: Optional[int] = None  # None = random seed


class StepRequest(BaseModel):
    action: Action


class LeaderboardSubmit(BaseModel):
    model_name: str
    score: float
    steps_taken: int
    root_cause_correct: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "SRE Incident Triage Environment",
        "version": "1.0.0",
        "tasks": ["task1", "task2", "task3", "task4"],
        "endpoints": ["/reset", "/step", "/state", "/leaderboard", "/health"],
    }


@app.get("/health")
def health():
    """Health check — used by HuggingFace Space ping."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = None):
    """
    Start a new episode. Generates a fresh incident from the given seed.
    If no seed provided, one is chosen randomly (but returned so results are reproducible).
    """
    global _active_engine

    if req is None:
        req = ResetRequest()

    valid_tasks = {"task1", "task2", "task3", "task4"}
    if req.task_id not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"task_id must be one of {valid_tasks}")

    seed = req.seed if req.seed is not None else random.randint(0, 999999)
    _active_engine = EpisodeEngine(task_id=req.task_id, seed=seed)

    return _active_engine.reset()


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """
    Take one action in the current episode.
    Must call /reset first to start an episode.
    """
    if _active_engine is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    return _active_engine.step(req.action)


@app.get("/state", response_model=EpisodeState)
def state():
    """
    Return the current episode state (for debugging, not for the agent).
    Includes step count, actions taken, and whether the episode is done.
    Note: does NOT expose the incident ground truth.
    """
    if _active_engine is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    s = _active_engine.get_state()
    # Strip ground truth from state before returning
    safe_state = EpisodeState(
        task_id=s.task_id,
        seed=s.seed,
        current_step=s.current_step,
        max_steps=s.max_steps,
        done=s.done,
        actions_taken=s.actions_taken,
        diagnosis_submitted=s.diagnosis_submitted,
        final_reward=s.final_reward,
    )
    return safe_state


@app.get("/leaderboard", response_model=List[LeaderboardEntry])
def leaderboard():
    """Return all completed episode scores, sorted by score descending."""
    return sorted(_leaderboard, key=lambda e: e.score, reverse=True)


@app.post("/leaderboard/submit")
def submit_to_leaderboard(req: LeaderboardSubmit):
    """Called by inference.py at end of episode to record the score."""
    if _active_engine is None:
        raise HTTPException(status_code=400, detail="No active episode.")

    state = _active_engine.get_state()
    entry = LeaderboardEntry(
        model_name=req.model_name,
        task_id=state.task_id,
        seed=state.seed,
        score=req.score,
        steps_taken=req.steps_taken,
        root_cause_correct=req.root_cause_correct,
        timestamp=datetime.utcnow().isoformat(),
    )
    _leaderboard.append(entry)
    return {"status": "recorded", "entry": entry}


@app.get("/tasks")
def list_tasks():
    """Describe all available tasks and their difficulty."""
    return {
        "tasks": [
            {
                "id": "task1",
                "name": "Single service failure",
                "difficulty": "easy",
                "max_steps": 6,
                "description": "One service has a clear failure. Logs and alerts point directly to root cause.",
            },
            {
                "id": "task2",
                "name": "Cascading failure",
                "difficulty": "medium",
                "max_steps": 8,
                "description": "A failure in one service causes degradation in a downstream service. "
                               "Agent must trace the cascade to the true root cause.",
            },
            {
                "id": "task3",
                "name": "Silent failure",
                "difficulty": "hard",
                "max_steps": 10,
                "description": "No critical alerts. Business metrics are dropping. "
                               "Agent must correlate logs, metrics, and a recent deploy to find the cause.",
            },
            {
                "id": "task4",
                "name": "Multi-service incident",
                "difficulty": "expert",
                "max_steps": 12,
                "description": "Three services show anomalies. Agent must investigate all of them, "
                               "identify the single root cause, run the correct remediation, "
                               "and submit a complete incident report.",
            },
        ]
    }
