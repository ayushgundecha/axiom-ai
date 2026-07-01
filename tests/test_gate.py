"""CI gate + reporting-only evaluator + gap_variance term (RR25)."""

from __future__ import annotations

import pytest

from axiom.config import AxiomSettings
from axiom.exceptions import EvaluationError
from axiom.robustness.gate import ReportingOnlyRobustness, reporting_evaluator, run_ci_gate
from axiom.robustness.labeler import RunLabel
from axiom.robustness.metrics import compute_rrs


def _run(version, agent, *, proxy_pass, oracle_pass, task="t"):
    return RunLabel(
        task_id=task,
        agent=agent,
        seed=1,
        reward_version=version,
        proxy_score=1.0 if proxy_pass else 0.0,
        proxy_pass=proxy_pass,
        oracle_score=1.0 if oracle_pass else 0.0,
        oracle_pass=oracle_pass,
        hack=(proxy_pass and not oracle_pass),
    )


def test_gate_passes_when_v1_is_robust():
    runs = [
        _run("v1", "exploiter", proxy_pass=False, oracle_pass=False),
        _run("v1", "honest", proxy_pass=True, oracle_pass=True),
    ]
    result = run_ci_gate(runs, min_macro_rrs=0.9, max_hack_rate=0.0)
    assert result.passed is True
    assert result.macro_rrs == 1.0
    assert result.reasons == []


def test_gate_fails_when_v1_leaks():
    runs = [
        _run("v1", "exploiter", proxy_pass=True, oracle_pass=False),  # leaked hack
        _run("v1", "honest", proxy_pass=True, oracle_pass=True),
    ]
    result = run_ci_gate(runs, min_macro_rrs=0.9, max_hack_rate=0.0)
    assert result.passed is False
    assert any("hack_rate" in r for r in result.reasons)


def test_gate_fails_when_no_runs_for_version():
    runs = [_run("v0", "exploiter", proxy_pass=True, oracle_pass=False)]
    result = run_ci_gate(runs, version="v1")
    assert result.passed is False


def test_reporting_only_evaluator_requires_flag():
    on = AxiomSettings(robustness_reporting_only=True)
    ev = reporting_evaluator(on)
    label = _run("v1", "honest", proxy_pass=True, oracle_pass=True)
    assert ev.reporting_only is True
    assert ev.gap(label) == 0.0
    with pytest.raises(EvaluationError, match="cannot be a live reward"):
        ev.as_live_reward(label)


def test_reporting_only_evaluator_rejects_disabled_flag():
    off = AxiomSettings(robustness_reporting_only=False)
    with pytest.raises(EvaluationError, match="must be True"):
        ReportingOnlyRobustness(off)


def test_gap_variance_term_lowers_rrs():
    runs = [
        _run("v1", "exploiter", proxy_pass=False, oracle_pass=False),
        _run("v1", "honest", proxy_pass=True, oracle_pass=True),
    ]
    # Inject a gap to create variance.
    runs[0] = RunLabel(
        task_id="t",
        agent="exploiter",
        seed=1,
        reward_version="v1",
        proxy_score=0.9,
        proxy_pass=False,
        oracle_score=0.0,
        oracle_pass=False,
        hack=False,
    )
    without = compute_rrs(runs, use_gap_variance=False)
    with_var = compute_rrs(runs, use_gap_variance=True)
    assert with_var.gap_variance > 0
    assert with_var.rrs <= without.rrs
