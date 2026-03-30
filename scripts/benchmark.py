#!/usr/bin/env python3
"""Benchmark runner — compare Claude agent vs Random agent.

Runs both agents across all tasks and produces a comparison table
showing that Claude significantly outperforms random. This is the
evaluation credibility signal Deeptune wants to see.

Usage:
    # Start servers first, then:
    python scripts/benchmark.py
    python scripts/benchmark.py --envs json cli        # subset of envs
    python scripts/benchmark.py --agent random          # single agent only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Task registry: (env, task_id) pairs to benchmark
ALL_TASKS: list[tuple[str, str]] = [
    ("json", "create_and_complete"),
    ("cli", "organize_files"),
    ("cli", "find_and_extract"),
]


async def run_agent(
    agent: Any,
    agent_name: str,
    base_url: str,
    env_name: str,
    task_id: str,
) -> dict[str, Any]:
    """Run a single agent on a single task, return results."""
    start = time.time()
    try:
        evaluation = await agent.run_episode(
            base_url=base_url,
            env_name=env_name,
            task_id=task_id,
            verbose=False,
        )
        elapsed = time.time() - start
        scores = evaluation.get("scores", {})
        return {
            "agent": agent_name,
            "env": env_name,
            "task": task_id,
            "completion": scores.get("completion", 0),
            "efficiency": scores.get("efficiency", 0),
            "accuracy": scores.get("accuracy", 0),
            "safety": scores.get("safety", 0),
            "steps": scores.get("total_steps", 0),
            "elapsed": round(elapsed, 1),
            "error": None,
        }
    except Exception as e:
        return {
            "agent": agent_name,
            "env": env_name,
            "task": task_id,
            "completion": 0,
            "efficiency": 0,
            "accuracy": 0,
            "safety": 0,
            "steps": 0,
            "elapsed": 0,
            "error": str(e)[:60],
        }


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 95)
    print("  axiom-ai Benchmark Results")
    print("=" * 95)

    header = (
        f"  {'Agent':<10} {'Env':<8} {'Task':<25} "
        f"{'Comp':>5} {'Eff':>5} {'Acc':>5} {'Safe':>5} "
        f"{'Steps':>5} {'Time':>6}"
    )
    print(header)
    print("-" * 95)

    for r in results:
        if r["error"]:
            print(f"  {r['agent']:<10} {r['env']:<8} {r['task']:<25}   ERROR: {r['error']}")
        else:
            print(
                f"  {r['agent']:<10} {r['env']:<8} {r['task']:<25} "
                f"{r['completion']:>5.2f} {r['efficiency']:>5.2f} "
                f"{r['accuracy']:>5.2f} {r['safety']:>5.2f} "
                f"{r['steps']:>5} {r['elapsed']:>5.1f}s"
            )

    print("-" * 95)

    # Aggregate by agent
    agents = sorted({r["agent"] for r in results})
    print("\n  Aggregate Scores:")
    for agent_name in agents:
        agent_results = [r for r in results if r["agent"] == agent_name and not r["error"]]
        if not agent_results:
            continue
        n = len(agent_results)
        avg_comp = sum(r["completion"] for r in agent_results) / n
        avg_eff = sum(r["efficiency"] for r in agent_results) / n
        avg_acc = sum(r["accuracy"] for r in agent_results) / n
        avg_safe = sum(r["safety"] for r in agent_results) / n

        bar = "█" * int(avg_comp * 20)
        print(
            f"    {agent_name:<10}: completion={avg_comp:.2f} {bar}  "
            f"efficiency={avg_eff:.2f}  accuracy={avg_acc:.2f}  "
            f"safety={avg_safe:.2f}"
        )

    print("=" * 95 + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="axiom-ai benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--envs", nargs="+", default=None, help="Filter by env names (e.g. --envs json cli)"
    )
    parser.add_argument(
        "--agent", default=None, choices=["claude", "random"], help="Run only one agent type"
    )
    args = parser.parse_args()

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

    print(f"\nRunning benchmark: {len(agents_to_run)} agent(s) x {len(tasks)} task(s)")

    results: list[dict[str, Any]] = []
    for agent_name, agent in agents_to_run:
        for env_name, task_id in tasks:
            print(f"  Running {agent_name} on {env_name}/{task_id}...", end=" ", flush=True)
            result = await run_agent(agent, agent_name, args.base_url, env_name, task_id)
            status = f"completion={result['completion']:.1f}" if not result["error"] else "ERROR"
            print(status)
            results.append(result)

    print_results_table(results)


if __name__ == "__main__":
    asyncio.run(main())
