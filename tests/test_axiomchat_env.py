"""Unit tests for axiom/envs/axiomchat_env.py — the AxiomChat (mini-Slack) env.

These tests do NOT require a running app or a browser; they exercise the
subclass's reset-body hook, URL defaulting, registration, and TaskConfig
seed/scale validation. The browser-driven episode lives in the integration
test (test_axiomchat_integration.py).
"""

from typing import Any

import pytest

from axiom.models import TaskConfig


def _bare_config(**overrides: Any) -> TaskConfig:
    base: dict[str, Any] = {
        "name": "t",
        "env": "axiomchat",
        "description": "d",
        "goal": {"type": "custom_js", "script": "return true;"},
    }
    base.update(overrides)
    return TaskConfig(**base)


class TestAxiomChatEnvironmentUnit:
    def test_env_id_is_axiomchat(self, sample_axiomchat_task_config: dict[str, Any]) -> None:
        from axiom.envs.axiomchat_env import AxiomChatEnvironment

        env = AxiomChatEnvironment(TaskConfig(**sample_axiomchat_task_config))
        assert env.env_id == "axiomchat"

    @pytest.mark.asyncio
    async def test_reset_server_passes_seed_and_scale(
        self, sample_axiomchat_task_config: dict[str, Any]
    ) -> None:
        from axiom.envs.axiomchat_env import AxiomChatEnvironment

        env = AxiomChatEnvironment(TaskConfig(**sample_axiomchat_task_config))
        body = await env._reset_server()
        assert body == {"seed": 42, "scale": "medium"}

    @pytest.mark.asyncio
    async def test_reset_server_defaults_when_unset(self) -> None:
        from axiom.envs.axiomchat_env import AxiomChatEnvironment

        env = AxiomChatEnvironment(_bare_config())
        body = await env._reset_server()
        assert body == {"seed": 1, "scale": "medium"}

    @pytest.mark.asyncio
    async def test_base_reset_server_is_empty(self) -> None:
        """The base WebAppEnvironment hook must stay an empty reset."""
        from axiom.envs.webapp_env import WebAppEnvironment

        env = WebAppEnvironment(
            TaskConfig(
                name="w",
                env="webapp",
                description="d",
                goal={"type": "custom_js", "script": "return true;"},
            )
        )
        assert await env._reset_server() == {}

    def test_app_url_defaults_to_settings(self) -> None:
        from axiom.config import get_settings
        from axiom.envs.axiomchat_env import AxiomChatEnvironment

        env = AxiomChatEnvironment(_bare_config())
        assert env._app_url == get_settings().axiomchat_app_url

    def test_app_url_respects_explicit_value(self) -> None:
        from axiom.envs.axiomchat_env import AxiomChatEnvironment

        env = AxiomChatEnvironment(_bare_config(app_url="http://chat.example:3100"))
        assert env._app_url == "http://chat.example:3100"


class TestAxiomChatRegistration:
    def test_registered_in_app_registry(self) -> None:
        from axiom.api.app import _create_registry

        registry = _create_registry()
        assert "axiomchat" in registry.list_envs()

    def test_registry_creates_axiomchat_instance(self) -> None:
        from axiom.api.app import _create_registry
        from axiom.envs.axiomchat_env import AxiomChatEnvironment

        registry = _create_registry()
        env = registry.create("axiomchat", _bare_config(seed=5))
        assert isinstance(env, AxiomChatEnvironment)
        assert env.env_id == "axiomchat"


class TestTaskConfigSeedScale:
    def test_seed_and_scale_validate(self) -> None:
        cfg = _bare_config(seed=7, scale="small")
        assert cfg.seed == 7
        assert cfg.scale == "small"

    def test_seed_and_scale_default_to_none(self) -> None:
        cfg = _bare_config()
        assert cfg.seed is None
        assert cfg.scale is None
