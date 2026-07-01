"""Best-of-N exploiter selection + concurrency (RR17)."""

from __future__ import annotations

import pytest

from axiom.robustness.best_of_n import best_of_n_hack_rate, run_best_of_n, select_best
from axiom.robustness.labeler import RunLabel


def _label(*, hack, proxy_score=0.0, oracle_score=0.0):
    return RunLabel(
        task_id="t",
        agent="exploiter",
        seed=1,
        reward_version="v0",
        proxy_score=proxy_score,
        proxy_pass=proxy_score >= 0.5,
        oracle_score=oracle_score,
        oracle_pass=oracle_score >= 0.5,
        hack=hack,
    )


def test_select_best_prefers_hack():
    labels = [
        _label(hack=False, proxy_score=0.9),
        _label(hack=True, proxy_score=0.5),
        _label(hack=False, proxy_score=0.3),
    ]
    assert select_best(labels).hack is True


def test_select_best_falls_back_to_highest_proxy():
    labels = [_label(hack=False, proxy_score=0.2), _label(hack=False, proxy_score=0.8)]
    assert select_best(labels).proxy_score == 0.8


def test_select_best_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        select_best([])


async def test_run_best_of_n_finds_the_hack():
    # Only attempt #2 produces a hack.
    async def attempt(i: int) -> RunLabel:
        return _label(hack=(i == 2), proxy_score=0.5 if i == 2 else 0.1)

    best = await run_best_of_n(attempt, n=4, concurrency=2)
    assert best.hack is True


async def test_run_best_of_n_validates_n():
    async def attempt(_i):
        return _label(hack=False)

    with pytest.raises(ValueError, match="n must be"):
        await run_best_of_n(attempt, n=0)


async def test_best_of_n_hack_rate_across_targets():
    # target A: a hack on attempt 1; target B: never hacks.
    async def target_a(i):
        return _label(hack=(i == 1), proxy_score=0.6 if i == 1 else 0.1)

    async def target_b(_i):
        return _label(hack=False, proxy_score=0.2)

    rate = await best_of_n_hack_rate([target_a, target_b], n=3)
    assert rate == 0.5
