"""Report builder — reports/robustness.json + leaderboard (RR15).

Assembles labeled runs into the self-contained JSON the leaderboard renders:

    {
      "generated_at", "seeds", "scale",
      "runs": [RunLabel, ...],
      "summary": {
        "by_task_version": {"task::version": {hack_rate, honest_fidelity, gap, rrs, ...}},
        "leaderboard": [{cell, task, reward_version, ...}, ...],  # sorted by rrs
        "macro_rrs": float
      },
      "judge_substudy": {...} | null
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axiom.robustness.labeler import RunLabel
from axiom.robustness.metrics import macro_rrs, rrs_by_task_version


def build_summary(
    runs: list[RunLabel],
    *,
    use_gap_variance: bool = False,
) -> dict[str, Any]:
    """Per (task, version) RRS, a sorted leaderboard, and the macro-RRS."""
    by = rrs_by_task_version(runs, use_gap_variance=use_gap_variance)
    by_task_version = {key: score.to_dict() for key, score in by.items()}

    leaderboard: list[dict[str, Any]] = []
    for cell, score in by.items():
        task, _, version = cell.partition("::")
        entry: dict[str, Any] = {"cell": cell, "task": task, "reward_version": version}
        entry.update(score.to_dict())
        leaderboard.append(entry)
    # Highest RRS first; ties broken by cell name for determinism.
    leaderboard.sort(key=lambda e: (-float(e["rrs"]), str(e["cell"])))

    return {
        "by_task_version": by_task_version,
        "leaderboard": leaderboard,
        "macro_rrs": macro_rrs(list(by.values())),
    }


def build_report(
    runs: list[RunLabel],
    *,
    generated_at: str,
    seeds: list[int],
    scale: str = "medium",
    judge_substudy: dict[str, Any] | None = None,
    seed_split: dict[str, Any] | None = None,
    use_gap_variance: bool = False,
) -> dict[str, Any]:
    """Build the full robustness report dict (JSON-serializable)."""
    return {
        "generated_at": generated_at,
        "seeds": seeds,
        "seed_split": seed_split,
        "scale": scale,
        "runs": [r.to_dict() for r in runs],
        "summary": build_summary(runs, use_gap_variance=use_gap_variance),
        "judge_substudy": judge_substudy,
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
    """Write the report dict to ``path`` as pretty JSON (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False))
    return path
