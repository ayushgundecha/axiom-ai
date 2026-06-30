"""Hardening methodology: named defenses reproduce + justify each v1 (RR14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core.task_loader import TaskLoader
from axiom.robustness.corpus import corpus_target_from_oracle, load_corpus
from axiom.robustness.hardening import (
    DEFENSES,
    defenses_for_task,
    harden_spec,
    hardening_report,
    merged_overrides,
)
from axiom.robustness.labeler import build_grading_inputs
from axiom.robustness.proxies import build_proxy
from tests.robustness_fixtures import apply_actions, make_oracle_state

REWARD_TASKS = [
    "answer_support_question",
    "summarize_incident",
    "assign_request",
    "triage_backlog",
]


def _loader() -> TaskLoader:
    return TaskLoader(Path(__file__).resolve().parent.parent / "tasks")


TASKS = {t: _loader().load_task("axiomchat", t) for t in REWARD_TASKS}


def test_defenses_catalog_well_formed():
    for name, d in DEFENSES.items():
        assert d.name == name
        assert d.proxy_class
        assert d.description.strip()
        assert isinstance(d.spec_overrides, dict) and d.spec_overrides


@pytest.mark.parametrize("task_id", REWARD_TASKS)
def test_authored_v1_is_v0_plus_named_defenses(task_id: str):
    """The authored v1 proxy contains exactly the overrides of its named defenses."""
    task = TASKS[task_id]
    assert task.proxy is not None
    v1 = task.proxy["v1"]
    overrides = merged_overrides(defenses_for_task(task_id))
    for key, value in overrides.items():
        assert v1.get(key) == value, f"{task_id} v1 missing defense override {key}={value}"


@pytest.mark.parametrize("task_id", REWARD_TASKS)
def test_harden_spec_reconstructs_authored_v1_behaviorally(task_id: str):
    task = TASKS[task_id]
    assert task.proxy is not None
    v0 = task.proxy["v0"]
    v1 = task.proxy["v1"]
    to_class = v1["type"] if v0["type"] != v1["type"] else None
    reconstructed = harden_spec(v0, defenses_for_task(task_id), to_class=to_class)
    # Every authored-v1 hardening key is present in the reconstruction.
    for key, value in v1.items():
        if key == "type":
            assert reconstructed["type"] == value
        else:
            assert reconstructed.get(key) == value


async def _proxy_pass(case, proxy) -> bool:
    pre = make_oracle_state()
    target = corpus_target_from_oracle(pre, case.scenario)
    assert target is not None
    post = apply_actions(pre, case.actions(target))
    ctx, *_ = build_grading_inputs(pre, post, case.scenario)
    return proxy.passed(await proxy.score(ctx, TASKS[case.task_id]))


@pytest.mark.parametrize("task_id", REWARD_TASKS)
async def test_reconstructed_v1_defeats_exploits_keeps_honest(task_id: str):
    task = TASKS[task_id]
    v0, v1 = task.proxy["v0"], task.proxy["v1"]
    to_class = v1["type"] if v0["type"] != v1["type"] else None
    spec = {"v1": harden_spec(v0, defenses_for_task(task_id), to_class=to_class)}
    proxy = build_proxy(spec, hardened=True)

    cases = [c for c in load_corpus() if c.task_id == task_id]
    for case in cases:
        passed = await _proxy_pass(case, proxy)
        if case.kind == "exploit":
            assert passed is False, f"{case.id} leaks through reconstructed v1"
        else:
            assert passed is True, f"honest {case.id} wrongly rejected (false negative)"


def test_hardening_report():
    report = hardening_report("assign_request")
    assert report.task_id == "assign_request"
    assert "identity-guard" in report.defenses
    assert report.overrides.get("forbid_author") is True
