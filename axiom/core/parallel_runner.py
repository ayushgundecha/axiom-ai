"""Parallel episode runner for concurrent agent evaluation.

Runs multiple agent episodes concurrently with bounded parallelism
using asyncio.Semaphore. Each episode is isolated — a failure in one
does not affect others.

At Deeptune scale, you need thousands of episodes running concurrently.
This module is the infrastructure for that.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class AgentProtocol(Protocol):
    """Minimal protocol for agents used in parallel runs."""

    async def run_episode(
        self,
        base_url: str,
        env_name: str,
        task_id: str,
        verbose: bool = ...,
    ) -> dict[str, Any]: ...


@dataclass
class EpisodeConfig:
    """Configuration for a single episode."""

    env_name: str
    task_id: str
    agent_name: str


@dataclass
class EpisodeResult:
    """Result from a single episode run."""

    env_name: str
    task_id: str
    agent_name: str
    scores: dict[str, float]
    elapsed_seconds: float
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Aggregate statistics across all episode results."""

    results: list[EpisodeResult] = field(default_factory=list)

    @property
    def successful(self) -> list[EpisodeResult]:
        """Results that completed without error."""
        return [r for r in self.results if r.error is None]

    @property
    def failed(self) -> list[EpisodeResult]:
        """Results that had errors."""
        return [r for r in self.results if r.error is not None]

    def agent_summary(self) -> dict[str, dict[str, float]]:
        """Per-agent aggregate stats: mean and stdev per metric."""
        agents: dict[str, list[EpisodeResult]] = {}
        for r in self.successful:
            agents.setdefault(r.agent_name, []).append(r)

        summary: dict[str, dict[str, float]] = {}
        for agent_name, agent_results in sorted(agents.items()):
            metrics: dict[str, float] = {}
            for key in ("completion", "efficiency", "accuracy", "safety"):
                values = [r.scores.get(key, 0.0) for r in agent_results]
                metrics[f"{key}_mean"] = (
                    round(statistics.mean(values), 3) if values else 0.0
                )
                metrics[f"{key}_std"] = (
                    round(statistics.stdev(values), 3)
                    if len(values) >= 2
                    else 0.0
                )
            metrics["episodes"] = float(len(agent_results))
            metrics["errors"] = float(
                sum(1 for r in self.results if r.agent_name == agent_name and r.error)
            )
            summary[agent_name] = metrics
        return summary


class ParallelRunner:
    """Run multiple episodes concurrently with bounded parallelism.

    Usage:
        runner = ParallelRunner(base_url="http://localhost:8000", max_concurrency=5)
        results = await runner.run_batch(episodes, agents)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_concurrency: int = 5,
    ) -> None:
        self._base_url = base_url
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run_episode(
        self,
        config: EpisodeConfig,
        agent: AgentProtocol,
        on_complete: Callable[[EpisodeResult], Awaitable[None] | None] | None = None,
    ) -> EpisodeResult:
        """Run a single episode with semaphore-limited concurrency."""
        async with self._semaphore:
            start = time.time()
            try:
                evaluation = await agent.run_episode(
                    base_url=self._base_url,
                    env_name=config.env_name,
                    task_id=config.task_id,
                    verbose=False,
                )
                elapsed = time.time() - start
                scores: dict[str, float] = {}
                raw_scores = evaluation.get("scores", {})
                for key in ("completion", "efficiency", "accuracy", "safety"):
                    scores[key] = float(raw_scores.get(key, 0.0))
                result = EpisodeResult(
                    env_name=config.env_name,
                    task_id=config.task_id,
                    agent_name=config.agent_name,
                    scores=scores,
                    elapsed_seconds=round(elapsed, 2),
                )
            except Exception as e:
                elapsed = time.time() - start
                result = EpisodeResult(
                    env_name=config.env_name,
                    task_id=config.task_id,
                    agent_name=config.agent_name,
                    scores={},
                    elapsed_seconds=round(elapsed, 2),
                    error=str(e)[:120],
                )

        if on_complete is not None:
            callback_result = on_complete(result)
            if asyncio.iscoroutine(callback_result):
                await callback_result

        return result

    async def run_batch(
        self,
        episodes: list[tuple[EpisodeConfig, AgentProtocol]],
        on_complete: Callable[[EpisodeResult], Awaitable[None] | None] | None = None,
    ) -> list[EpisodeResult]:
        """Run all episodes concurrently (bounded by semaphore)."""
        tasks = [
            self.run_episode(config, agent, on_complete)
            for config, agent in episodes
        ]
        return list(await asyncio.gather(*tasks))

    async def run_benchmark(
        self,
        agents: list[tuple[str, AgentProtocol]],
        task_list: list[tuple[str, str]],
        on_complete: Callable[[EpisodeResult], Awaitable[None] | None] | None = None,
    ) -> BenchmarkReport:
        """Run all agent x task combinations and return a report."""
        episodes: list[tuple[EpisodeConfig, AgentProtocol]] = [
            (
                EpisodeConfig(env_name=env, task_id=task, agent_name=name),
                agent,
            )
            for name, agent in agents
            for env, task in task_list
        ]
        results = await self.run_batch(episodes, on_complete)
        return BenchmarkReport(results=results)
