"""Trajectory recording and persistence.

Records every step of an episode (action, observation, reward) and saves
trajectories to disk. Screenshots are saved as separate PNG files —
never stored as inline base64 in JSON. This is how real training data
pipelines work: you don't want 50MB base64 strings in trajectory files.

Output structure:
    trajectories/
        {session_id}/
            trajectory.json    # Metadata + steps (no base64)
            screenshots/
                step_1.png
                step_2.png
                ...
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from axiom.models import Trajectory, TrajectoryStep


class TrajectoryRecorder:
    """Records and persists episode trajectories."""

    def __init__(self) -> None:
        self._trajectories: dict[str, Trajectory] = {}

    def start_session(
        self,
        session_id: str,
        task_name: str,
        env_type: str,
    ) -> None:
        """Initialize trajectory recording for a session."""
        self._trajectories[session_id] = Trajectory(
            session_id=session_id,
            task_name=task_name,
            env_type=env_type,
        )

    def record_step(
        self,
        session_id: str,
        step_num: int,
        action: dict[str, Any],
        observation: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record a single step in the trajectory."""
        trajectory = self._trajectories.get(session_id)
        if trajectory is None:
            msg = f"No trajectory started for session '{session_id}'"
            raise ValueError(msg)

        step = TrajectoryStep(
            step_num=step_num,
            action=action,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        trajectory.steps.append(step)

    def set_evaluation(
        self,
        session_id: str,
        evaluation: dict[str, Any],
    ) -> None:
        """Attach evaluation scores to a trajectory."""
        trajectory = self.get_trajectory(session_id)
        trajectory.evaluation = evaluation

    def get_trajectory(self, session_id: str) -> Trajectory:
        """Retrieve the trajectory for a session."""
        trajectory = self._trajectories.get(session_id)
        if trajectory is None:
            msg = f"No trajectory for session '{session_id}'"
            raise ValueError(msg)
        return trajectory

    def save(self, session_id: str, trajectory_dir: Path) -> Path:
        """Save trajectory to disk.

        Screenshots are extracted from observations and saved as separate
        PNG files. The trajectory JSON contains no base64 data.

        Returns the directory path where files were saved.
        """
        trajectory = self.get_trajectory(session_id)
        session_dir = trajectory_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Extract screenshots and clean observations for JSON serialization
        screenshots_dir = session_dir / "screenshots"
        serializable_steps: list[dict[str, Any]] = []

        for step in trajectory.steps:
            step_dict = step.model_dump()
            obs = step_dict.get("observation", {})

            # Extract and save screenshot as separate file
            screenshot_b64 = obs.pop("screenshot_base64", None)
            if screenshot_b64:
                if not screenshots_dir.exists():
                    screenshots_dir.mkdir(parents=True, exist_ok=True)
                png_path = screenshots_dir / f"step_{step.step_num}.png"
                png_path.write_bytes(base64.b64decode(screenshot_b64))

            serializable_steps.append(step_dict)

        # Write trajectory JSON (no base64)
        trajectory_data: dict[str, Any] = {
            "session_id": trajectory.session_id,
            "task_name": trajectory.task_name,
            "env_type": trajectory.env_type,
            "total_steps": len(trajectory.steps),
            "steps": serializable_steps,
        }

        if trajectory.evaluation:
            trajectory_data["evaluation"] = trajectory.evaluation

        trajectory_file = session_dir / "trajectory.json"
        trajectory_file.write_text(json.dumps(trajectory_data, indent=2))

        return session_dir
