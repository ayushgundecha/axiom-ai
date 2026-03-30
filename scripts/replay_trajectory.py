#!/usr/bin/env python3
"""Replay a saved trajectory step-by-step.

Loads a trajectory JSON file and prints each step with action,
reward, and cumulative progress.

Usage:
    python scripts/replay_trajectory.py trajectories/{session_id}/trajectory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def replay(trajectory_path: str) -> None:
    path = Path(trajectory_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    data = json.loads(path.read_text())

    print(f"\n{'=' * 60}")
    print("  Trajectory Replay")
    print(f"  Session: {data.get('session_id', '?')}")
    print(f"  Task:    {data.get('task_name', '?')}")
    print(f"  Env:     {data.get('env_type', '?')}")
    print(f"  Steps:   {data.get('total_steps', '?')}")
    print(f"{'=' * 60}\n")

    total_reward = 0.0
    for step in data.get("steps", []):
        action = step.get("action", {})
        reward = step.get("reward", 0)
        terminated = step.get("terminated", False)
        truncated = step.get("truncated", False)
        total_reward += reward

        action_str = f"{action.get('type', '?')}"
        if action.get("selector"):
            action_str += f" -> {action['selector']}"
        if action.get("value"):
            action_str += f" = '{str(action['value'])[:40]}'"

        status = "DONE" if terminated else ("TIMEOUT" if truncated else "->")
        print(
            f"  Step {step.get('step_num', '?'):>2}: {action_str:<50} "
            f"reward={reward:+.2f}  total={total_reward:.2f}  {status}"
        )

    # Check for screenshots
    screenshots_dir = path.parent / "screenshots"
    if screenshots_dir.exists():
        count = len(list(screenshots_dir.glob("*.png")))
        print(f"\n  Screenshots: {count} saved in {screenshots_dir}/")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay axiom trajectory")
    parser.add_argument("trajectory", help="Path to trajectory.json")
    args = parser.parse_args()
    replay(args.trajectory)
