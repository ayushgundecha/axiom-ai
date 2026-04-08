#!/usr/bin/env python3
"""Parallel benchmark runner — evaluate agents concurrently.

Runs multiple agent x task combinations in parallel using asyncio,
with bounded concurrency. Produces a comparison table with per-agent
aggregate statistics (mean, stdev).

Usage:
    python scripts/parallel_benchmark.py
    python scripts/parallel_benchmark.py --concurrency 3
    python scripts/parallel_benchmark.py --envs json cli --agent random
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.core.parallel_runner import (  # noqa: E402
    BenchmarkReport,
    EpisodeConfig,
    EpisodeResult,
    ParallelRunner,
)

ALL_TASKS: list[tuple[str, str]] = [
    ("json", "create_and_complete"),
    ("cli", "organize_files"),
    ("cli", "find_and_extract"),
]


def _load_dotenv(dotenv_path: Path) -> None:
    """Load KEY=VALUE pairs from .env into os.environ."""
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def print_results(report: BenchmarkReport) -> None:
    """Print formatted results table and aggregate summary."""
    print(f"\n{'=' * 100}")
    print("  axiom-ai Parallel Benchmark Results")
    print(f"{'=' * 100}")

    header = (
        f"  {'Agent':<10} {'Env':<8} {'Task':<25} "
        f"{'Comp':>5} {'Eff':>5} {'Acc':>5} {'Safe':>5} "
        f"{'Time':>6}"
    )
    print(header)
    print(f"{'-' * 100}")

    for r in report.results:
        if r.error:
            print(
                f"  {r.agent_name:<10} {r.env_name:<8} {r.task_id:<25}"
                f"   ERROR: {r.error[:50]}"
            )
        else:
            print(
                f"  {r.agent_name:<10} {r.env_name:<8} {r.task_id:<25} "
                f"{r.scores.get('completion', 0):>5.2f} "
                f"{r.scores.get('efficiency', 0):>5.2f} "
                f"{r.scores.get('accuracy', 0):>5.2f} "
                f"{r.scores.get('safety', 0):>5.2f} "
                f"{r.elapsed_seconds:>5.1f}s"
            )

    print(f"{'-' * 100}")

    # Aggregate summary
    summary = report.agent_summary()
    if summary:
        print("\n  Aggregate Scores (mean +/- stdev):")
        for agent_name, metrics in summary.items():
            n = int(metrics["episodes"])
            errs = int(metrics["errors"])
            comp = metrics["completion_mean"]
            comp_std = metrics["completion_std"]
            bar = "█" * int(comp * 20)

            print(
                f"    {agent_name:<10}: "
                f"completion={comp:.2f}±{comp_std:.2f} {bar}  "
                f"efficiency={metrics['efficiency_mean']:.2f}  "
                f"accuracy={metrics['accuracy_mean']:.2f}  "
                f"safety={metrics['safety_mean']:.2f}  "
                f"({n} ok, {errs} err)"
            )

    total = len(report.results)
    ok = len(report.successful)
    fail = len(report.failed)
    print(f"\n  Total: {total} episodes  |  {ok} succeeded  |  {fail} failed")
    print(f"{'=' * 100}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="axiom-ai parallel benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Max concurrent episodes (default: 5)"
    )
    parser.add_argument(
        "--envs", nargs="+", default=None, help="Filter by env names"
    )
    parser.add_argument(
        "--agent", default=None, choices=["claude", "random"], help="Run only one agent"
    )
    args = parser.parse_args()

    _load_dotenv(Path(__file__).parent.parent / ".env")

    tasks = ALL_TASKS
    if args.envs:
        tasks = [(e, t) for e, t in tasks if e in args.envs]

    if not tasks:
        print("No tasks match the filter.")
        return

    agents_to_run: list[tuple[str, Any]] = []

    if args.agent != "claude":
        from agents.random_agent import RandomAgent
        agents_to_run.append(("Random", RandomAgent()))

    if args.agent != "random":
        try:
            from agents.claude_agent import ClaudeAgent
            agents_to_run.append(("Claude", ClaudeAgent()))
        except Exception:
            print("  (Skipping Claude — ANTHROPIC_API_KEY not set)")

    if not agents_to_run:
        print("No agents available to run.")
        return

    total = len(agents_to_run) * len(tasks)
    completed = 0
    print(
        f"\nRunning parallel benchmark: {len(agents_to_run)} agent(s) x "
        f"{len(tasks)} task(s) = {total} episodes (concurrency={args.concurrency})"
    )

    def on_complete(result: EpisodeResult) -> None:
        nonlocal completed
        completed += 1
        status = (
            f"completion={result.scores.get('completion', 0):.1f}"
            if not result.error
            else f"ERROR: {result.error[:40]}"
        )
        print(f"  [{completed}/{total}] {result.agent_name} on {result.env_name}/{result.task_id}: {status}")

    runner = ParallelRunner(
        base_url=args.base_url,
        max_concurrency=args.concurrency,
    )
    report = await runner.run_benchmark(agents_to_run, tasks, on_complete)
    print_results(report)


if __name__ == "__main__":
    asyncio.run(main())
