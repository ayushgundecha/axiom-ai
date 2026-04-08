"""Tests for LLM-as-judge evaluator and composite evaluator.

All tests mock the Anthropic client — no real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axiom.core.evaluator import CompositeEvaluator
from axiom.exceptions import EvaluationError
from axiom.models import EvaluationResult, Observation, Trajectory, TrajectoryStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_env(
    step_count: int = 3,
    max_steps: int = 10,
    optimal_steps: int = 2,
) -> MagicMock:
    """Create a mock BaseEnvironment with observation and evaluate."""
    env = MagicMock()
    env.step_count = step_count
    env.max_steps = max_steps
    env.task_config = MagicMock()
    env.task_config.optimal_steps = optimal_steps

    obs = Observation(
        task_description="Write a README",
        available_action_types=["run_command"],
        step_count=step_count,
        max_steps=max_steps,
        text_output="README.md created successfully",
    )
    env.observe = AsyncMock(return_value=obs)
    env.evaluate = AsyncMock(
        return_value={
            "completion": 1.0,
            "efficiency": 0.8,
            "accuracy": 1.0,
            "safety": 1.0,
            "total_steps": step_count,
            "optimal_steps": optimal_steps,
            "invalid_actions": 0,
        }
    )
    return env


def _make_trajectory() -> Trajectory:
    return Trajectory(
        session_id="test_sess",
        task_name="write_readme",
        env_type="cli",
        steps=[
            TrajectoryStep(
                step_num=1,
                action={"type": "run_command", "value": "cat main.py"},
                observation={"text_output": "def hello(): ..."},
                reward=0.05,
                terminated=False,
                truncated=False,
            ),
            TrajectoryStep(
                step_num=2,
                action={"type": "run_command", "value": "echo '# Project' > README.md"},
                observation={"text_output": ""},
                reward=1.05,
                terminated=True,
                truncated=False,
            ),
        ],
    )


def _mock_claude_response(scores_json: str) -> MagicMock:
    """Create a mock Anthropic messages.create response."""
    content_block = MagicMock()
    content_block.text = scores_json
    response = MagicMock()
    response.content = [content_block]
    return response


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator tests
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluator:
    """LLM judge must call Claude API and return structured EvaluationResult."""

    @pytest.mark.asyncio
    async def test_returns_evaluation_result(self) -> None:
        from axiom.core.llm_judge import LLMJudgeEvaluator

        env = _make_mock_env()
        judge = LLMJudgeEvaluator(rubric={"completion": "Did it finish?"})
        judge.set_trajectory(_make_trajectory())

        mock_response = _mock_claude_response(
            '{"completion": 0.9, "efficiency": 0.7, "accuracy": 0.85, "safety": 1.0}'
        )

        with patch.object(judge, "_client") as mock_client:
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            result = await judge.evaluate(env)

        assert isinstance(result, EvaluationResult)
        assert result.completion == 0.9
        assert result.efficiency == 0.7
        assert result.accuracy == 0.85
        assert result.safety == 1.0
        assert result.total_steps == 3
        assert result.optimal_steps == 2

    @pytest.mark.asyncio
    async def test_includes_rubric_in_prompt(self) -> None:
        from axiom.core.llm_judge import LLMJudgeEvaluator

        env = _make_mock_env()
        rubric: dict[str, Any] = {
            "completion": "All sections present?",
            "accuracy": "Professional quality?",
        }
        judge = LLMJudgeEvaluator(rubric=rubric)

        mock_response = _mock_claude_response(
            '{"completion": 1.0, "efficiency": 1.0, "accuracy": 1.0, "safety": 1.0}'
        )

        with patch.object(judge, "_client") as mock_client:
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            await judge.evaluate(env)

            # Check the prompt includes rubric text
            call_kwargs = mock_client.messages.create.call_args
            user_message: str = call_kwargs.kwargs["messages"][0]["content"]
            assert "All sections present?" in user_message
            assert "Professional quality?" in user_message

    @pytest.mark.asyncio
    async def test_handles_api_error(self) -> None:
        import anthropic

        from axiom.core.llm_judge import LLMJudgeEvaluator

        env = _make_mock_env()
        judge = LLMJudgeEvaluator()

        with patch.object(judge, "_client") as mock_client:
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=anthropic.APIError(
                    message="rate limited",
                    request=MagicMock(),
                    body=None,
                )
            )

            with pytest.raises(EvaluationError, match="LLM judge API call failed"):
                await judge.evaluate(env)

    @pytest.mark.asyncio
    async def test_clamps_scores_to_valid_range(self) -> None:
        from axiom.core.llm_judge import LLMJudgeEvaluator

        env = _make_mock_env()
        judge = LLMJudgeEvaluator()

        # Claude returns out-of-range scores
        mock_response = _mock_claude_response(
            '{"completion": 1.5, "efficiency": -0.3, "accuracy": 0.5, "safety": 2.0}'
        )

        with patch.object(judge, "_client") as mock_client:
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            result = await judge.evaluate(env)

        assert result.completion == 1.0
        assert result.efficiency == 0.0
        assert result.accuracy == 0.5
        assert result.safety == 1.0

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self) -> None:
        from axiom.core.llm_judge import LLMJudgeEvaluator

        env = _make_mock_env()
        judge = LLMJudgeEvaluator()

        mock_response = _mock_claude_response(
            '```json\n{"completion": 0.8, "efficiency": 0.6, "accuracy": 0.9, "safety": 1.0}\n```'
        )

        with patch.object(judge, "_client") as mock_client:
            mock_client.messages = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            result = await judge.evaluate(env)

        assert result.completion == 0.8


# ---------------------------------------------------------------------------
# CompositeEvaluator tests
# ---------------------------------------------------------------------------


class TestCompositeEvaluator:
    """Composite must weighted-average float scores."""

    @pytest.mark.asyncio
    async def test_weighted_average(self) -> None:
        eval_a = MagicMock()
        eval_a.evaluate = AsyncMock(
            return_value=EvaluationResult(
                completion=1.0, efficiency=0.8, accuracy=1.0, safety=1.0,
                total_steps=3, optimal_steps=2, invalid_actions=0,
            )
        )
        eval_b = MagicMock()
        eval_b.evaluate = AsyncMock(
            return_value=EvaluationResult(
                completion=0.5, efficiency=0.6, accuracy=0.5, safety=0.8,
                total_steps=3, optimal_steps=2, invalid_actions=0,
            )
        )

        composite = CompositeEvaluator([(0.6, eval_a), (0.4, eval_b)])
        env = _make_mock_env()
        result = await composite.evaluate(env)

        # 0.6*1.0 + 0.4*0.5 = 0.8
        assert result.completion == 0.8
        # 0.6*0.8 + 0.4*0.6 = 0.72
        assert result.efficiency == 0.72
        # 0.6*1.0 + 0.4*0.5 = 0.8
        assert result.accuracy == 0.8
        # 0.6*1.0 + 0.4*0.8 = 0.92
        assert result.safety == 0.92

    @pytest.mark.asyncio
    async def test_preserves_integer_fields_from_first(self) -> None:
        eval_a = MagicMock()
        eval_a.evaluate = AsyncMock(
            return_value=EvaluationResult(
                completion=1.0, efficiency=1.0, accuracy=1.0, safety=1.0,
                total_steps=5, optimal_steps=3, invalid_actions=1,
            )
        )
        eval_b = MagicMock()
        eval_b.evaluate = AsyncMock(
            return_value=EvaluationResult(
                completion=1.0, efficiency=1.0, accuracy=1.0, safety=1.0,
                total_steps=99, optimal_steps=99, invalid_actions=99,
            )
        )

        composite = CompositeEvaluator([(0.5, eval_a), (0.5, eval_b)])
        result = await composite.evaluate(_make_mock_env())

        # Integer fields come from first evaluator
        assert result.total_steps == 5
        assert result.optimal_steps == 3
        assert result.invalid_actions == 1

    def test_empty_evaluators_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one evaluator"):
            CompositeEvaluator([])

    @pytest.mark.asyncio
    async def test_single_evaluator_passthrough(self) -> None:
        """With one evaluator at weight 1.0, result should pass through."""
        single = MagicMock()
        single.evaluate = AsyncMock(
            return_value=EvaluationResult(
                completion=0.75, efficiency=0.9, accuracy=0.8, safety=1.0,
                total_steps=4, optimal_steps=3, invalid_actions=0,
            )
        )
        composite = CompositeEvaluator([(1.0, single)])
        result = await composite.evaluate(_make_mock_env())

        assert result.completion == 0.75
        assert result.efficiency == 0.9
