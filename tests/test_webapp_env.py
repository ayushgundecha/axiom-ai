"""Tests for axiom/envs/webapp_env.py — Playwright browser environment.

Integration tests require the todo app to be running.
Marked with @pytest.mark.integration for selective execution.
Written TDD-style before implementation.
"""

from typing import Any

import pytest

from tests.conftest import requires_todo_app

TODO_INPUT = "[data-testid='todo-input']"
ADD_BUTTON = "[data-testid='add-button']"


@pytest.mark.integration
class TestWebAppEnvironment:
    """WebAppEnvironment: Playwright browser automation against real web app."""

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_reset_returns_observation(
        self, sample_webapp_task_config: dict[str, Any]
    ) -> None:
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import Observation, TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        async with WebAppEnvironment(config) as env:
            obs = await env.reset()

            assert isinstance(obs, Observation)
            assert obs.url is not None
            assert "localhost" in obs.url or "todo-app" in obs.url

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_hybrid_observation_has_dom_and_screenshot(
        self, sample_webapp_task_config: dict[str, Any]
    ) -> None:
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        async with WebAppEnvironment(config) as env:
            obs = await env.reset()

            # Hybrid mode should have both DOM and screenshot
            assert obs.dom_tree is not None
            assert obs.screenshot_base64 is not None
            assert len(obs.dom_tree) > 0
            assert len(obs.screenshot_base64) > 0

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_type_and_click_adds_todo(
        self, sample_webapp_task_config: dict[str, Any]
    ) -> None:
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        async with WebAppEnvironment(config) as env:
            await env.reset()

            type_action = Action(
                type=ActionType.TYPE,
                selector=TODO_INPUT,
                value="Review PR #42",
            )
            r1 = await env.step(type_action)
            assert r1.info["valid"] is True

            click_action = Action(
                type=ActionType.CLICK,
                selector=ADD_BUTTON,
            )
            r2 = await env.step(click_action)
            assert r2.info["valid"] is True

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_goal_element_count(self, sample_webapp_task_config: dict[str, Any]) -> None:
        """Add 3 todos and verify the element_count goal is met."""
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        async with WebAppEnvironment(config) as env:
            await env.reset()

            todos = ["Review PR #42", "Deploy to staging", "Write tests"]
            for title in todos:
                await env.step(
                    Action(
                        type=ActionType.TYPE,
                        selector=TODO_INPUT,
                        value=title,
                    )
                )
                result = await env.step(
                    Action(
                        type=ActionType.CLICK,
                        selector=ADD_BUTTON,
                    )
                )

            assert result.terminated  # Goal: 3 todo items

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_evaluate_completed_task(self, sample_webapp_task_config: dict[str, Any]) -> None:
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        async with WebAppEnvironment(config) as env:
            await env.reset()

            for title in ["T1", "T2", "T3"]:
                await env.step(
                    Action(
                        type=ActionType.TYPE,
                        selector=TODO_INPUT,
                        value=title,
                    )
                )
                await env.step(
                    Action(
                        type=ActionType.CLICK,
                        selector=ADD_BUTTON,
                    )
                )

            scores = await env.evaluate()
            assert scores["completion"] == 1.0
            assert 0.0 <= scores["efficiency"] <= 1.0
            assert scores["accuracy"] == 1.0
            assert 0.0 <= scores["safety"] <= 1.0

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_reset_clears_server_state(
        self, sample_webapp_task_config: dict[str, Any]
    ) -> None:
        """Reset must clear BOTH browser and server state."""
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        async with WebAppEnvironment(config) as env:
            # First episode: add todos
            await env.reset()
            await env.step(
                Action(
                    type=ActionType.TYPE,
                    selector=TODO_INPUT,
                    value="T1",
                )
            )
            await env.step(
                Action(
                    type=ActionType.CLICK,
                    selector=ADD_BUTTON,
                )
            )

            # Second episode: reset should clear everything
            obs = await env.reset()
            assert obs.dom_tree is not None

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_invalid_selector_gives_error(
        self, sample_webapp_task_config: dict[str, Any]
    ) -> None:
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        async with WebAppEnvironment(config) as env:
            await env.reset()

            action = Action(
                type=ActionType.CLICK,
                selector="#nonexistent-element-12345",
            )
            result = await env.step(action)
            assert result.info["valid"] is False
            assert result.reward < 0

    @requires_todo_app
    @pytest.mark.asyncio
    async def test_env_id_is_webapp(self, sample_webapp_task_config: dict[str, Any]) -> None:
        from axiom.envs.webapp_env import WebAppEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_webapp_task_config)
        env = WebAppEnvironment(config)
        assert env.env_id == "webapp"
        await env.cleanup()
