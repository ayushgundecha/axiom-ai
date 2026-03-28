"""Tests for axiom/envs/json_env.py — JSON state machine environment.

First concrete environment. Validates the entire core framework works end-to-end.
Written TDD-style before implementation.
"""

from typing import Any

import pytest


class TestJSONEnvironment:
    """JSONEnvironment: pure Python state machine for baseline testing."""

    @pytest.mark.asyncio
    async def test_reset_returns_observation(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Observation, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            obs = await env.reset()

            assert isinstance(obs, Observation)
            assert obs.state is not None
            assert obs.state["todos"] == []
            assert obs.step_count == 0
            assert obs.task_description == config.description

    @pytest.mark.asyncio
    async def test_step_add_todo(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Action, ActionType, StepResult, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            await env.reset()

            action = Action(
                type=ActionType.API_CALL,
                value="add_todo",
                params={"title": "Buy milk"},
            )
            result = await env.step(action)

            assert isinstance(result, StepResult)
            assert result.observation.state is not None
            assert len(result.observation.state["todos"]) == 1
            assert result.observation.state["todos"][0]["title"] == "Buy milk"
            assert result.reward > 0  # Small positive reward for valid action

    @pytest.mark.asyncio
    async def test_step_complete_todo(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            await env.reset()

            # Add a todo first
            add = Action(type=ActionType.API_CALL, value="add_todo", params={"title": "Task 1"})
            result = await env.step(add)
            todo_id = result.observation.state["todos"][0]["id"]

            # Complete it
            complete = Action(
                type=ActionType.API_CALL,
                value="complete_todo",
                params={"id": todo_id},
            )
            result = await env.step(complete)
            assert result.observation.state["todos"][0]["completed"] is True

    @pytest.mark.asyncio
    async def test_goal_met_terminates(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            await env.reset()

            # Goal requires 2 todos
            a1 = Action(type=ActionType.API_CALL, value="add_todo", params={"title": "T1"})
            r1 = await env.step(a1)
            assert not r1.terminated

            a2 = Action(type=ActionType.API_CALL, value="add_todo", params={"title": "T2"})
            r2 = await env.step(a2)
            assert r2.terminated  # Goal met: 2 todos
            assert r2.reward >= 1.0  # Big reward for completion

    @pytest.mark.asyncio
    async def test_max_steps_truncates(self) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(
            name="truncation_test",
            env="json",
            description="Test truncation.",
            max_steps=2,
            goal={"type": "element_count", "field": "todos", "count": 100},
        )
        async with JSONEnvironment(config) as env:
            await env.reset()

            action = Action(type=ActionType.API_CALL, value="add_todo", params={"title": "X"})
            await env.step(action)
            result = await env.step(action)

            assert result.truncated  # Hit max steps
            assert not result.terminated  # Goal not met

    @pytest.mark.asyncio
    async def test_evaluate_returns_scores(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            await env.reset()

            # Complete the goal
            a1 = Action(type=ActionType.API_CALL, value="add_todo", params={"title": "T1"})
            a2 = Action(type=ActionType.API_CALL, value="add_todo", params={"title": "T2"})
            await env.step(a1)
            await env.step(a2)

            scores = await env.evaluate()
            assert scores["completion"] == 1.0
            assert 0.0 <= scores["efficiency"] <= 1.0
            assert scores["accuracy"] == 1.0
            assert 0.0 <= scores["safety"] <= 1.0

    @pytest.mark.asyncio
    async def test_step_without_reset_raises(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.exceptions import EnvironmentNotReady
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        env = JSONEnvironment(config)

        action = Action(type=ActionType.API_CALL, value="add_todo", params={"title": "T"})
        with pytest.raises(EnvironmentNotReady):
            await env.step(action)

        await env.cleanup()

    @pytest.mark.asyncio
    async def test_invalid_action_gives_negative_reward(
        self, sample_json_task_config: dict[str, Any]
    ) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            await env.reset()

            # Invalid operation
            action = Action(type=ActionType.API_CALL, value="nonexistent_op", params={})
            result = await env.step(action)
            assert result.reward < 0

    @pytest.mark.asyncio
    async def test_observe_without_stepping(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import Observation, TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            await env.reset()
            obs = await env.observe()

            assert isinstance(obs, Observation)
            assert obs.state["todos"] == []

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_json_task_config)
        env = JSONEnvironment(config)
        await env.reset()
        await env.cleanup()
        await env.cleanup()  # Should not raise

    @pytest.mark.asyncio
    async def test_async_context_manager(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_json_task_config)
        async with JSONEnvironment(config) as env:
            obs = await env.reset()
            assert obs is not None
        # After exiting context, env should be cleaned up

    @pytest.mark.asyncio
    async def test_env_id_is_json(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.envs.json_env import JSONEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_json_task_config)
        env = JSONEnvironment(config)
        assert env.env_id == "json"
        await env.cleanup()
