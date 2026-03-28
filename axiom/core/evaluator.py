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
