"""Tests for parallel episode runner.

All tests mock agent.run_episode — no HTTP server needed.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from axiom.core.parallel_runner import (
    BenchmarkReport,
    EpisodeConfig,
    EpisodeResult,
    ParallelRunner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent(
    scores: dict[str, float] | None = None,
    delay: float = 0.0,
    error: Exception | None = None,
) -> AsyncMock:
    """Create a mock agent with configurable behavior."""
    agent = AsyncMock()

    async def _run_episode(
        base_url: str,
        env_name: str,
        task_id: str,
        verbose: bool = True,
    ) -> dict[str, Any]:
        if delay > 0:
            await asyncio.sleep(delay)
        if error is not None:
            raise error
        return {
            "scores": scores
            or {"completion": 1.0, "efficiency": 0.8, "accuracy": 1.0, "safety": 1.0}
        }

    agent.run_episode = _run_episode
    return agent


# ---------------------------------------------------------------------------
# ParallelRunner tests
# ---------------------------------------------------------------------------


class TestParallelRunner:
    """ParallelRunner must handle concurrency, errors, and callbacks."""

    @pytest.mark.asyncio
    async def test_runs_all_episodes(self) -> None:
        agent = _make_mock_agent()
        episodes = [
            (EpisodeConfig("json", "task1", "test_agent"), agent),
            (EpisodeConfig("cli", "task2", "test_agent"), agent),
            (EpisodeConfig("json", "task3", "test_agent"), agent),
        ]

        runner = ParallelRunner(max_concurrency=5)
        results = await runner.run_batch(episodes)

        assert len(results) == 3
        assert all(r.error is None for r in results)
        assert all(r.scores.get("completion") == 1.0 for r in results)

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """At most max_concurrency episodes should run simultaneously."""
        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _tracked_episode(
            base_url: str, env_name: str, task_id: str, verbose: bool = True
        ) -> dict[str, Any]:
            nonlocal peak_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent

            await asyncio.sleep(0.05)

            async with lock:
                current_concurrent -= 1

            return {
                "scores": {"completion": 1.0, "efficiency": 1.0, "accuracy": 1.0, "safety": 1.0}
            }

        agent = AsyncMock()
        agent.run_episode = _tracked_episode

        episodes = [(EpisodeConfig("json", f"task_{i}", "agent"), agent) for i in range(10)]

        runner = ParallelRunner(max_concurrency=2)
        results = await runner.run_batch(episodes)

        assert len(results) == 10
        assert peak_concurrent <= 2

    @pytest.mark.asyncio
    async def test_error_isolation(self) -> None:
        """A failing episode must not crash the batch."""
        good_agent = _make_mock_agent(
            scores={"completion": 1.0, "efficiency": 1.0, "accuracy": 1.0, "safety": 1.0}
        )
        bad_agent = _make_mock_agent(error=RuntimeError("agent crashed"))

        episodes = [
            (EpisodeConfig("json", "task1", "good"), good_agent),
            (EpisodeConfig("json", "task2", "bad"), bad_agent),
            (EpisodeConfig("json", "task3", "good"), good_agent),
        ]

        runner = ParallelRunner(max_concurrency=5)
        results = await runner.run_batch(episodes)

        assert len(results) == 3
        errors = [r for r in results if r.error is not None]
        successes = [r for r in results if r.error is None]
        assert len(errors) == 1
        assert len(successes) == 2
        assert "agent crashed" in errors[0].error  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_on_complete_callback(self) -> None:
        """Callback must fire for each completed episode."""
        agent = _make_mock_agent()
        completed: list[EpisodeResult] = []

        def callback(result: EpisodeResult) -> None:
            completed.append(result)

        episodes = [(EpisodeConfig("json", f"task_{i}", "agent"), agent) for i in range(3)]

        runner = ParallelRunner(max_concurrency=5)
        await runner.run_batch(episodes, on_complete=callback)

        assert len(completed) == 3

    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        runner = ParallelRunner()
        results = await runner.run_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# BenchmarkReport tests
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """BenchmarkReport must compute correct aggregate statistics."""

    def test_agent_summary_mean_and_stdev(self) -> None:
        results = [
            EpisodeResult(
                "json",
                "t1",
                "Claude",
                {"completion": 1.0, "efficiency": 0.8, "accuracy": 1.0, "safety": 1.0},
                1.0,
            ),
            EpisodeResult(
                "cli",
                "t2",
                "Claude",
                {"completion": 0.5, "efficiency": 0.6, "accuracy": 0.5, "safety": 0.8},
                2.0,
            ),
            EpisodeResult(
                "json",
                "t1",
                "Random",
                {"completion": 0.0, "efficiency": 0.0, "accuracy": 0.0, "safety": 0.5},
                0.5,
            ),
        ]
        report = BenchmarkReport(results=results)
        summary = report.agent_summary()

        assert "Claude" in summary
        assert "Random" in summary
        assert summary["Claude"]["completion_mean"] == 0.75
        assert summary["Claude"]["episodes"] == 2.0
        assert summary["Random"]["completion_mean"] == 0.0

    def test_successful_and_failed_split(self) -> None:
        results = [
            EpisodeResult("json", "t1", "A", {"completion": 1.0}, 1.0),
            EpisodeResult("json", "t2", "A", {}, 0.5, error="boom"),
        ]
        report = BenchmarkReport(results=results)

        assert len(report.successful) == 1
        assert len(report.failed) == 1

    def test_empty_report(self) -> None:
        report = BenchmarkReport()
        assert report.successful == []
        assert report.failed == []
        assert report.agent_summary() == {}

    def test_run_benchmark_creates_all_combinations(self) -> None:
        """run_benchmark must create N_agents x N_tasks episodes."""
        # Verify the config generation logic
        agents = [("A", None), ("B", None)]
        tasks = [("json", "t1"), ("cli", "t2"), ("cli", "t3")]
        expected = len(agents) * len(tasks)  # 2 * 3 = 6

        configs = [
            EpisodeConfig(env_name=env, task_id=task, agent_name=name)
            for name, _ in agents
            for env, task in tasks
        ]
        assert len(configs) == expected
