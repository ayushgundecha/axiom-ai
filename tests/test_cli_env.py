"""Tests for axiom/envs/cli_env.py — CLI/terminal environment.

Tests sandboxing, safety, command execution, and goal checking.
Written TDD-style before implementation.
"""

from pathlib import Path
from typing import Any

import pytest


class TestCLIEnvironment:
    """CLIEnvironment: sandboxed shell command execution."""

    @pytest.mark.asyncio
    async def test_reset_creates_sandbox(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Observation, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            obs = await env.reset()

            assert isinstance(obs, Observation)
            assert obs.text_output is not None
            assert "readme.md" in obs.text_output
            assert "notes.txt" in obs.text_output

    @pytest.mark.asyncio
    async def test_run_allowed_command(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            action = Action(type=ActionType.RUN_COMMAND, value="ls")
            result = await env.step(action)

            assert result.info["valid"] is True
            assert result.reward > 0

    @pytest.mark.asyncio
    async def test_blocked_command_rejected(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            # 'python' is not in the allowed commands list
            action = Action(type=ActionType.RUN_COMMAND, value="python -c 'import os'")
            result = await env.step(action)

            assert result.info["valid"] is False
            assert result.reward < 0

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            action = Action(type=ActionType.RUN_COMMAND, value="cat ../../etc/passwd")
            result = await env.step(action)

            assert result.info["valid"] is False

    @pytest.mark.asyncio
    async def test_mkdir_and_mv(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            mkdir = Action(type=ActionType.RUN_COMMAND, value="mkdir docs")
            await env.step(mkdir)

            mv = Action(type=ActionType.RUN_COMMAND, value="mv readme.md docs/")
            result = await env.step(mv)

            assert result.info["valid"] is True

    @pytest.mark.asyncio
    async def test_goal_directory_structure(self, sample_cli_task_config: dict[str, Any]) -> None:
        """Full task: create docs/ and move readme.md into it."""
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            actions = [
                Action(type=ActionType.RUN_COMMAND, value="mkdir docs"),
                Action(type=ActionType.RUN_COMMAND, value="mv readme.md docs/"),
            ]
            for action in actions:
                result = await env.step(action)

            # Goal: docs/readme.md exists
            assert result.terminated

    @pytest.mark.asyncio
    async def test_evaluate_completed_task(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            actions = [
                Action(type=ActionType.RUN_COMMAND, value="mkdir docs"),
                Action(type=ActionType.RUN_COMMAND, value="mv readme.md docs/"),
            ]
            for action in actions:
                await env.step(action)

            scores = await env.evaluate()
            assert scores["completion"] == 1.0
            assert scores["safety"] > 0.0

    @pytest.mark.asyncio
    async def test_non_run_command_rejected(self, sample_cli_task_config: dict[str, Any]) -> None:
        """CLI env only accepts RUN_COMMAND actions."""
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            action = Action(type=ActionType.CLICK, selector="#btn")
            result = await env.step(action)

            assert result.info["valid"] is False

    @pytest.mark.asyncio
    async def test_command_timeout(self, sample_cli_task_config: dict[str, Any]) -> None:
        """Commands that take too long should be killed."""
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        async with CLIEnvironment(config) as env:
            await env.reset()

            # sleep is not in allowed commands, so this tests the safety check
            action = Action(type=ActionType.RUN_COMMAND, value="sleep 100")
            result = await env.step(action)
            assert result.info["valid"] is False

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_dir(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        env = CLIEnvironment(config)
        await env.reset()
        workdir = Path(env._workdir)
        assert workdir.exists()  # noqa: ASYNC240

        await env.cleanup()
        assert not workdir.exists()  # noqa: ASYNC240

    @pytest.mark.asyncio
    async def test_step_without_reset_raises(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.exceptions import EnvironmentNotReady
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        env = CLIEnvironment(config)

        action = Action(type=ActionType.RUN_COMMAND, value="ls")
        with pytest.raises(EnvironmentNotReady):
            await env.step(action)

        await env.cleanup()

    @pytest.mark.asyncio
    async def test_env_id_is_cli(self, sample_cli_task_config: dict[str, Any]) -> None:
        from axiom.envs.cli_env import CLIEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_cli_task_config)
        env = CLIEnvironment(config)
        assert env.env_id == "cli"
        await env.cleanup()
