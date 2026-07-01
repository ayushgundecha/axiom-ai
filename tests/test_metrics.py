"""RRS metric on synthetic labeled runs (RR11)."""

from __future__ import annotations

from axiom.robustness.labeler import RunLabel
from axiom.robustness.metrics import compute_rrs, macro_rrs, rrs_by_task_version


def _run(
    agent,
    *,
    proxy_pass,
    oracle_pass,
    proxy_score=None,
    oracle_score=None,
    version="v0",
    task="t",
    error=None,
):
    ps = proxy_score if proxy_score is not None else (1.0 if proxy_pass else 0.0)
    os_ = oracle_score if oracle_score is not None else (1.0 if oracle_pass else 0.0)
    return RunLabel(
        task_id=task,
        agent=agent,
        seed=1,
        reward_version=version,
        proxy_score=ps,
        proxy_pass=proxy_pass,
        oracle_score=os_,
        oracle_pass=oracle_pass,
        hack=(proxy_pass and not oracle_pass),
        error=error,
    )


def test_naive_proxy_high_hack_rate_low_rrs():
    runs = [
        _run("exploiter", proxy_pass=True, oracle_pass=False),  # hack
        _run("exploiter", proxy_pass=True, oracle_pass=False),  # hack
        _run("honest", proxy_pass=True, oracle_pass=True),
    ]
    score = compute_rrs(runs)
    assert score.hack_rate == 1.0
    assert score.honest_fidelity == 1.0
    assert score.rrs == 0.0  # (1 - 1.0) * 1.0


def test_hardened_proxy_zero_hack_rate_high_rrs():
    runs = [
        _run("exploiter", proxy_pass=False, oracle_pass=False),  # defeated
        _run("exploiter", proxy_pass=False, oracle_pass=False),
        _run("honest", proxy_pass=True, oracle_pass=True),
        _run("honest", proxy_pass=True, oracle_pass=True),
    ]
    score = compute_rrs(runs)
    assert score.hack_rate == 0.0
    assert score.honest_fidelity == 1.0
    assert score.rrs == 1.0


def test_honest_fidelity_only_over_oracle_true_honest_runs():
    runs = [
        _run("honest", proxy_pass=True, oracle_pass=True),
        _run("honest", proxy_pass=False, oracle_pass=True),  # honest but proxy missed it
        _run("honest", proxy_pass=False, oracle_pass=False),  # not oracle-true -> excluded
    ]
    score = compute_rrs(runs)
    assert score.n_honest == 2  # only oracle-true honest runs count
    assert score.honest_fidelity == 0.5


def test_gap_and_variance():
    runs = [
        _run("exploiter", proxy_pass=True, oracle_pass=False, proxy_score=1.0, oracle_score=0.0),
        _run("honest", proxy_pass=True, oracle_pass=True, proxy_score=1.0, oracle_score=1.0),
    ]
    score = compute_rrs(runs, use_gap_variance=True)
    assert score.proxy_oracle_gap == 0.5  # mean of (1.0, 0.0)
    assert score.gap_variance == 0.25
    # rrs = (1-1.0)*1.0 * (1-0.25) = 0
    assert score.rrs == 0.0


def test_no_exploit_runs_means_zero_hack_rate():
    runs = [_run("honest", proxy_pass=True, oracle_pass=True)]
    score = compute_rrs(runs)
    assert score.hack_rate == 0.0
    assert score.n_exploit == 0
    assert score.rrs == 1.0


def test_errors_ignored():
    runs = [
        _run("exploiter", proxy_pass=True, oracle_pass=False, error="boom"),
        _run("honest", proxy_pass=True, oracle_pass=True),
    ]
    score = compute_rrs(runs)
    assert score.n_exploit == 0  # errored run ignored
    assert score.hack_rate == 0.0


def test_rrs_by_task_version_and_macro():
    runs = [
        _run("exploiter", proxy_pass=True, oracle_pass=False, version="v0"),
        _run("honest", proxy_pass=True, oracle_pass=True, version="v0"),
        _run("exploiter", proxy_pass=False, oracle_pass=False, version="v1"),
        _run("honest", proxy_pass=True, oracle_pass=True, version="v1"),
    ]
    by = rrs_by_task_version(runs)
    assert set(by) == {"t::v0", "t::v1"}
    assert by["t::v0"].rrs == 0.0
    assert by["t::v1"].rrs == 1.0
    assert macro_rrs(list(by.values())) == 0.5
