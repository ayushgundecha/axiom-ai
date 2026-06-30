#!/usr/bin/env python3
"""Automated hacker-fixer loop (RR24, P2 STRETCH).

The manual hardening methodology (RR14), automated. Starting from each task's
naive (v0) proxy, repeatedly:

  1. HACK   — run the scripted exploit corpus and label it (proxy vs oracle);
  2. detect — collect the exploits that still hack (proxy_pass ∧ ¬oracle_pass);
  3. FIX    — for each hacking exploit, look up the named defense(s) that defeat
              it (axiom/robustness/hardening.py) and apply them to the proxy;
  4. repeat until hack_rate == 0 (or no further fix is available).

Deterministic + offline (uses the in-memory simulator) — no LLM, no Docker.
This is the optional stretch; core hardening stays manual.

Usage:
    python scripts/hack_fix_loop.py
    python scripts/hack_fix_loop.py --tasks answer_support_question
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.core.task_loader import TaskLoader  # noqa: E402
from axiom.robustness.corpus import RewardCase, corpus_target_from_oracle, load_corpus  # noqa: E402
from axiom.robustness.hardening import Defense, defenses_that_defeat  # noqa: E402
from axiom.robustness.labeler import label_episode  # noqa: E402
from axiom.robustness.oracles import build_oracle  # noqa: E402
from axiom.robustness.proxies import build_proxy  # noqa: E402
from axiom.robustness.simulator import apply_actions, build_workspace  # noqa: E402

ALL_TASKS = [
    "answer_support_question",
    "summarize_incident",
    "assign_request",
    "triage_backlog",
]


async def _hack_label(task, spec: dict, case: RewardCase, seed: int) -> bool:
    """Return True if ``case`` is a hack (proxy_pass ∧ ¬oracle_pass) under ``spec``."""
    base = build_workspace(seed=seed)
    target = corpus_target_from_oracle(base, case.scenario)
    if target is None:
        return False
    post = apply_actions(base, case.actions(target))
    # build_proxy selects the v0 variant from the spec block.
    proxy = build_proxy({"v0": spec}, hardened=False)
    oracle = build_oracle(task.oracle or {})
    label = await label_episode(
        None, None, task, proxy, oracle, base, post, agent=case.kind, seed=seed
    )
    return label.hack


async def harden_task(task, cases: list[RewardCase], seeds: list[int], max_rounds: int = 8):
    """Run the hacker-fixer loop for one task; return (final_spec, applied, rounds)."""
    spec = dict(task.proxy["v0"])  # start naive
    exploit_cases = [c for c in cases if c.kind == "exploit"]
    applied: list[str] = []

    for round_num in range(1, max_rounds + 1):
        hacking: list[RewardCase] = []
        for case in exploit_cases:
            if any([await _hack_label(task, spec, case, s) for s in seeds]):
                hacking.append(case)
        if not hacking:
            print(f"  round {round_num}: 0 hacks remaining ✓")
            return spec, applied, round_num
        print(f"  round {round_num}: {len(hacking)} exploit(s) still hacking: "
              f"{sorted({c.source for c in hacking})}")

        # FIX: apply the named defenses that defeat the still-hacking exploits.
        fixes: list[Defense] = []
        for case in hacking:
            fixes += defenses_that_defeat(case.source)
        new = False
        for d in fixes:
            if d.name not in applied:
                spec = {**spec, **d.spec_overrides, "type": "dom_regex"}
                applied.append(d.name)
                new = True
                print(f"    + applied defense: {d.name} ({d.description})")
        if not new:
            print("    no further defense available — stopping")
            return spec, applied, round_num

    return spec, applied, max_rounds


async def main() -> None:
    parser = argparse.ArgumentParser(description="automated hacker-fixer loop")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    loader = TaskLoader(Path(__file__).parent.parent / "tasks")
    cases = load_corpus()
    print(f"\nHacker-fixer loop (offline) over seeds {args.seeds}\n{'=' * 60}")

    for task_id in args.tasks:
        task = loader.load_task("axiomchat", task_id)
        task_cases = [c for c in cases if c.task_id == task_id]
        print(f"\n{task_id}:")
        spec, applied, rounds = await harden_task(task, task_cases, args.seeds)
        print(f"  -> hardened in {rounds} round(s); defenses: {applied}")

    print(f"\n{'=' * 60}\nDone. Compare with the manually-authored v1 specs in tasks/axiomchat/.\n")


if __name__ == "__main__":
    asyncio.run(main())
