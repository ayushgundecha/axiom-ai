"""Trajectory browsing endpoints for the replay UI."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter

from axiom.config import get_settings

router = APIRouter()


@router.get("/trajectories/saved")
async def list_saved_trajectories() -> list[dict[str, Any]]:
    """List all saved trajectories on disk.

    Scans the trajectory directory for session subdirectories containing
    trajectory.json files. Returns metadata for each (no step data).
    """
    settings = get_settings()
    trajectory_dir = settings.trajectory_dir
    trajectories: list[dict[str, Any]] = []

    if not trajectory_dir.exists():
        return trajectories

    for session_dir in sorted(trajectory_dir.iterdir()):
        trajectory_file = session_dir / "trajectory.json"
        if not trajectory_file.exists():
            continue
        try:
            data = json.loads(trajectory_file.read_text())
            trajectories.append({
                "session_id": data.get("session_id", session_dir.name),
                "task_name": data.get("task_name", "unknown"),
                "env_type": data.get("env_type", "unknown"),
                "total_steps": data.get("total_steps", 0),
                "has_evaluation": "evaluation" in data,
            })
        except (json.JSONDecodeError, OSError):
            continue

    return trajectories
