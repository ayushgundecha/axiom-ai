"""Best-of-N exploiter (RR17).

A single exploiter attempt under-estimates how gameable a proxy is. A stronger
adversary samples N attempts (varied prompts / temperatures / seeds) and keeps
the best — the one that actually achieves a hack, or, failing that, the one that
got closest. Running the N attempts concurrently mirrors
:class:`axiom.core.parallel_runner.ParallelRunner` (bounded by a semaphore).

This module is the selection + concurrency harness; the per-attempt work is an
injected async callable, so it is deterministic and unit-testable (and works
identically whether attempts are LLM episodes or scripted).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from axiom.robustness.labeler import RunLabel

AttemptFn = Callable[[int], Awaitable[RunLabel]]


def select_best(labels: list[RunLabel]) -> RunLabel:
    """Pick the strongest exploiter attempt.

    A hack wins outright; among non-hacks the highest proxy_score (closest to
    slipping past the proxy) wins. Ties break toward the lower oracle_score
    (further from the truth = a more dangerous near-miss).
    """
    if not labels:
        msg = "select_best requires at least one attempt"
        raise ValueError(msg)
    return max(labels, key=lambda r: (r.hack, r.proxy_score, -r.oracle_score))


async def run_best_of_n(
    attempt_fn: AttemptFn,
    n: int,
    *,
    concurrency: int = 5,
) -> RunLabel:
    """Run ``n`` exploiter attempts concurrently and return the best one."""
    if n < 1:
        msg = "n must be >= 1"
        raise ValueError(msg)
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> RunLabel:
        async with sem:
            return await attempt_fn(i)

    results = await asyncio.gather(*(one(i) for i in range(n)))
    return select_best(list(results))


async def best_of_n_hack_rate(
    attempt_fns: list[AttemptFn],
    n: int,
    *,
    concurrency: int = 5,
) -> float:
    """Hack rate of a best-of-N exploiter across several targets.

    Each entry in ``attempt_fns`` is a target's attempt generator; best-of-N is
    taken per target, and the fraction of targets where the best attempt is a
    hack is returned.
    """
    if not attempt_fns:
        return 0.0
    bests = await asyncio.gather(
        *(run_best_of_n(fn, n, concurrency=concurrency) for fn in attempt_fns)
    )
    return round(sum(1 for b in bests if b.hack) / len(bests), 4)
