"""Evaluator protocol and default implementation.

Defines a protocol so evaluation strategy can be swapped independently
of environments (e.g. LLM-as-judge, composite evaluators). The default
implementation simply delegates to env.evaluate().
"""

from __future__ import annotations

from typing import Protocol

from axiom.models import EvaluationResult

from .base_env import BaseEnvironment


class Evaluator(Protocol):
    """Protocol for environment evaluators.

    Any class with an async evaluate() method satisfies this protocol.
    """

    async def evaluate(self, env: BaseEnvironment) -> EvaluationResult: ...


class DefaultEvaluator:
    """Default evaluator — delegates to the environment's own evaluate().

    Wraps the raw dict from env.evaluate() into a typed EvaluationResult.
    """

    async def evaluate(self, env: BaseEnvironment) -> EvaluationResult:
        """Evaluate environment state and return typed result."""
        scores = await env.evaluate()
        return EvaluationResult(**scores)


class CompositeEvaluator:
    """Weighted combination of multiple evaluators.

    Runs all evaluators in sequence and produces a weighted average of
    their float scores. Integer fields (total_steps, optimal_steps,
    invalid_actions) are taken from the first evaluator's result.

    Example:
        composite = CompositeEvaluator([
            (0.6, DefaultEvaluator()),
            (0.4, LLMJudgeEvaluator(rubric=...)),
        ])
        result = await composite.evaluate(env)
    """

    def __init__(self, evaluators: list[tuple[float, Evaluator]]) -> None:
        if not evaluators:
            msg = "CompositeEvaluator requires at least one evaluator"
            raise ValueError(msg)
        self._evaluators = evaluators

    async def evaluate(self, env: BaseEnvironment) -> EvaluationResult:
        """Run all evaluators and weighted-average their scores."""
        results: list[tuple[float, EvaluationResult]] = []
        for weight, evaluator in self._evaluators:
            result = await evaluator.evaluate(env)
            results.append((weight, result))

        total_weight = sum(w for w, _ in results)
        if total_weight == 0:
            total_weight = 1.0

        def _weighted_avg(field: str) -> float:
            val = sum(
                w * float(getattr(r, field)) for w, r in results
            ) / total_weight
            return round(max(0.0, min(1.0, val)), 3)

        # Integer fields from the first evaluator (rule-based source of truth)
        first = results[0][1]

        return EvaluationResult(
            completion=_weighted_avg("completion"),
            efficiency=_weighted_avg("efficiency"),
            accuracy=_weighted_avg("accuracy"),
            safety=_weighted_avg("safety"),
            total_steps=first.total_steps,
            optimal_steps=first.optimal_steps,
            invalid_actions=first.invalid_actions,
        )
