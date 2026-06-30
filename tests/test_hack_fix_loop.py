"""Smoke test for the automated hacker-fixer loop (RR24, stretch)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from axiom.robustness.hardening import defenses_that_defeat

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "hack_fix_loop.py"


def _load():
    spec = importlib.util.spec_from_file_location("hack_fix_loop", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LOOP = _load()


def test_defenses_that_defeat_maps_exploits():
    assert any(
        d.name == "mention-cardinality-guard" for d in defenses_that_defeat("mention-everyone")
    )
    assert any(d.name == "identity-guard" for d in defenses_that_defeat("dm-self"))
    assert defenses_that_defeat("not-a-real-exploit") == []


@pytest.mark.parametrize(
    "task_id",
    ["answer_support_question", "summarize_incident", "assign_request", "triage_backlog"],
)
async def test_loop_converges_to_zero_hacks(task_id: str):
    task = LOOP.TaskLoader(_ROOT / "tasks").load_task("axiomchat", task_id)
    cases = [c for c in LOOP.load_corpus() if c.task_id == task_id]
    spec, applied, rounds = await LOOP.harden_task(task, cases, seeds=[1, 2])

    # The loop applied real named defenses and converged.
    assert applied, "loop should apply at least one defense"
    # Verify the discovered spec leaves no exploit hacking on the eval seeds.
    for case in cases:
        if case.kind != "exploit":
            continue
        hacks = [await LOOP._hack_label(task, spec, case, s) for s in (1, 2)]
        assert not any(hacks), f"{case.id} still hacks after the loop"
