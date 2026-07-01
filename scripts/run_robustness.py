#!/usr/bin/env python3
"""Reward Robustness harness — the headline checkpoint.

Runs the matrix of reward tasks × reward-versions (v0 naive | v1 hardened) ×
agents (exploiter | honest) × seeds, labels every episode with the (gameable)
proxy and the (deterministic, privileged) oracle, computes the Reward Robustness
Score, prints an RRS table, and writes reports/robustness.json.

Load-bearing: the oracle token is held HERE (harness-side) only — it is never
passed to an agent or environment. The oracle is consulted strictly out-of-band,
AFTER each episode.

Two modes:
  * --offline (default): drive the deterministic in-memory AxiomChat simulator —
    no Docker, no LLM. Always demoable; this is the no-LLM P0 checkpoint.
  * --live: drive a running AxiomChat (:3100) through a real browser.

Usage:
    python scripts/run_robustness.py                       # offline, all tasks
    python scripts/run_robustness.py --train-seeds 1 2 3 --reward-versions v0 v1
    python scripts/run_robustness.py --live --tasks answer_support_question
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.config import get_settings  # noqa: E402
from axiom.core.task_loader import TaskLoader  # noqa: E402
from axiom.models import TaskConfig  # noqa: E402
from axiom.robustness.corpus import RewardCase, corpus_target_from_oracle, load_corpus  # noqa: E402
from axiom.robustness.labeler import RunLabel, label_episode  # noqa: E402
from axiom.robustness.metrics import macro_rrs, rrs_by_task_version  # noqa: E402
from axiom.robustness.oracles import build_oracle  # noqa: E402
from axiom.robustness.proxies import build_proxy  # noqa: E402
from axiom.robustness.report import build_report, write_report  # noqa: E402
from axiom.robustness.seeds import make_split, validate_split  # noqa: E402
from axiom.robustness.simulator import apply_actions, build_workspace  # noqa: E402

ALL_TASKS = [
    "answer_support_question",
    "summarize_incident",
    "assign_request",
    "triage_backlog",
]


def _load_tasks(tasks_dir: Path, task_ids: list[str]) -> dict[str, TaskConfig]:
    loader = TaskLoader(tasks_dir)
    return {tid: loader.load_task("axiomchat", tid) for tid in task_ids}


async def _label_case(
    task: TaskConfig,
    case: RewardCase,
    pre: dict,
    post: dict,
    *,
    seed: int,
    version: str,
) -> RunLabel:
    proxy = build_proxy(task.proxy or {}, hardened=(version == "v1"))
    oracle = build_oracle(task.oracle or {})
    return await label_episode(
        None, None, task, proxy, oracle, pre, post,
        agent=case.kind, seed=seed, reward_version=version,
    )


async def run_offline(
    tasks: dict[str, TaskConfig],
    cases: list[RewardCase],
    seeds: list[int],
    versions: list[str],
) -> list[RunLabel]:
    """Drive the deterministic in-memory simulator (no server, no LLM)."""
    labels: list[RunLabel] = []
    for seed in seeds:
        base = build_workspace(seed=seed)
        for case in cases:
            target = corpus_target_from_oracle(base, case.scenario)
            if target is None:
                continue  # scenario absent for this seed (e.g. triage)
            post = apply_actions(base, case.actions(target))
            for version in versions:
                labels.append(
                    await _label_case(tasks[case.task_id], case, base, post, seed=seed, version=version)
                )
    return labels


async def run_live(
    tasks: dict[str, TaskConfig],
    cases: list[RewardCase],
    seeds: list[int],
    versions: list[str],
) -> list[RunLabel]:
    """Drive a live AxiomChat through a real browser; fetch pre/post oracle state."""
    from axiom.envs.axiomchat_env import AxiomChatEnvironment
    from axiom.robustness.oracle_client import fetch_oracle_state

    settings = get_settings()
    app_url, token = settings.axiomchat_app_url, settings.axiomchat_oracle_token
    labels: list[RunLabel] = []
    for seed in seeds:
        for case in cases:
            task = tasks[case.task_id]
            cfg = task.model_copy(update={"seed": seed})
            async with AxiomChatEnvironment(cfg) as env:
                await env.reset()
                pre = await fetch_oracle_state(app_url, token)
                target = corpus_target_from_oracle(pre, case.scenario)
                if target is None:
                    continue
                for action in case.actions(target):
                    await env.step(action)
                post = await fetch_oracle_state(app_url, token)
                for version in versions:
                    labels.append(
                        await _label_case(task, case, pre, post, seed=seed, version=version)
                    )
    return labels


def print_rrs_table(labels: list[RunLabel], title: str) -> None:
    by = rrs_by_task_version(labels)
    print(f"\n{'=' * 92}")
    print(f"  Reward Robustness Score — {title}")
    print(f"{'=' * 92}")
    header = (
        f"  {'Task':<26} {'Ver':<4} {'#exp':>5} {'#hon':>5} "
        f"{'hack_rate':>10} {'honest_fid':>11} {'gap':>7} {'RRS':>7}"
    )
    print(header)
    print(f"  {'-' * 88}")
    for cell, score in by.items():
        task, _, version = cell.partition("::")
        bar = "█" * int(score.rrs * 12)
        print(
            f"  {task:<26} {version:<4} {score.n_exploit:>5} {score.n_honest:>5} "
            f"{score.hack_rate:>10.3f} {score.honest_fidelity:>11.3f} "
            f"{score.proxy_oracle_gap:>7.3f} {score.rrs:>7.3f} {bar}"
        )
    print(f"  {'-' * 88}")

    for version in ("v0", "v1"):
        cells = [s for k, s in by.items() if k.endswith(f"::{version}")]
        if cells:
            print(
                f"  macro-RRS [{version}] = {macro_rrs(cells):.3f}   "
                f"(mean hack_rate = {sum(c.hack_rate for c in cells) / len(cells):.3f})"
            )
    print(f"{'=' * 92}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="axiom reward-robustness harness")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=None,
                        help="held-out seeds for reporting (defaults to train-seeds)")
    parser.add_argument("--agents", nargs="+", default=["exploiter", "honest"],
                        choices=["exploiter", "honest"])
    parser.add_argument("--reward-versions", nargs="+", default=["v0", "v1"],
                        choices=["v0", "v1"])
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--judge", action="store_true", help="(P1) include the LLM judge sub-study")
    parser.add_argument("--live", action="store_true", help="drive a live AxiomChat instead of offline")
    parser.add_argument("--out", default="reports/robustness.json")
    args = parser.parse_args()

    split = make_split(args.train_seeds, args.eval_seeds)
    validate_split(list(split.train), list(split.eval), strict=False)
    eval_seeds = list(split.eval)
    if split.held_out:
        print(f"  held-out split: train={list(split.train)} eval={eval_seeds}")
    tasks_dir = Path(__file__).parent.parent / "tasks"
    tasks = _load_tasks(tasks_dir, args.tasks)

    want_exploit = "exploiter" in args.agents
    want_honest = "honest" in args.agents
    cases = [
        c
        for c in load_corpus()
        if c.task_id in args.tasks
        and ((c.kind == "exploit" and want_exploit) or (c.kind == "honest" and want_honest))
    ]

    runner = run_live if args.live else run_offline
    mode = "live AxiomChat" if args.live else "offline simulator"
    print(
        f"\nRunning robustness harness ({mode}): {len(args.tasks)} task(s) × "
        f"{len(args.reward_versions)} version(s) × {len(cases)} case(s) × "
        f"{len(eval_seeds)} eval-seed(s)"
    )

    labels = await runner(tasks, cases, eval_seeds, args.reward_versions)
    print_rrs_table(labels, title=mode)

    judge_substudy = None
    if args.judge:
        from axiom.robustness.judge_reward import default_anthropic_backend
        from axiom.robustness.judge_substudy import run_judge_substudy

        backend = default_anthropic_backend("claude-haiku-4-5-20251001") if args.live else None
        judge_substudy = await run_judge_substudy(backend=backend)
        s = judge_substudy["summary"]
        print(
            f"  judge sub-study: naive_fooled_rate={s['naive_fooled_rate']} "
            f"hardened_resisted_rate={s['hardened_resisted_rate']}\n"
        )

    report = build_report(
        labels,
        generated_at=datetime.now(timezone.utc).isoformat(),
        seeds=eval_seeds,
        scale="medium",
        judge_substudy=judge_substudy,
        seed_split=split.to_dict(),
    )
    out_path = Path(args.out)
    write_report(report, out_path)
    print(f"  wrote {out_path}  ({len(labels)} labeled runs)\n")


if __name__ == "__main__":
    asyncio.run(main())
