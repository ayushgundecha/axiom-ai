"""Tests for axiom/core/ — framework machinery.

Tests the registry, session manager, task loader, and trajectory recorder.
Written TDD-style before implementation.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestEnvironmentRegistry:
    """Registry must support registration, creation, and listing."""

    def test_register_and_list(self) -> None:
        from axiom.core.registry import EnvironmentRegistry

        registry = EnvironmentRegistry()
        mock_cls = MagicMock()
        registry.register("test_env", mock_cls)

        assert "test_env" in registry.list_envs()

    def test_create_returns_instance(self) -> None:
        from axiom.core.registry import EnvironmentRegistry
        from axiom.models import TaskConfig

        registry = EnvironmentRegistry()

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        registry.register("test_env", mock_cls)

        config = TaskConfig(
            name="t", env="test_env", description="d", max_steps=5,
            goal={"type": "element_count", "field": "x", "count": 1},
        )
        env = registry.create("test_env", config)
        assert env is mock_instance

    def test_create_unknown_env_raises(self) -> None:
        from axiom.core.registry import EnvironmentRegistry
        from axiom.exceptions import TaskConfigError
        from axiom.models import TaskConfig

        registry = EnvironmentRegistry()
        config = TaskConfig(
            name="t", env="nonexistent", description="d", max_steps=5,
            goal={"type": "element_count", "field": "x", "count": 1},
        )
        with pytest.raises(TaskConfigError):
            registry.create("nonexistent", config)

    def test_register_env_decorator(self) -> None:
        from axiom.core.registry import EnvironmentRegistry

        registry = EnvironmentRegistry()

        @registry.register_decorator("decorated_env")
        class FakeEnv:
            pass

        assert "decorated_env" in registry.list_envs()
        assert registry.get_env_class("decorated_env") is FakeEnv


# ---------------------------------------------------------------------------
# Session manager tests
# ---------------------------------------------------------------------------


class TestSessionManager:
    """SessionManager must manage session lifecycle with proper cleanup."""

    @pytest.mark.asyncio
    async def test_create_session(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.core.session import SessionManager

        mock_registry = MagicMock()
        mock_env = AsyncMock()
        mock_env.reset = AsyncMock()
        mock_env.observe = AsyncMock()
        mock_registry.create.return_value = mock_env

        manager = SessionManager(registry=mock_registry)
        session = await manager.create_session("json", "test_task", sample_json_task_config)

        assert session.session_id is not None
        assert len(session.session_id) == 12  # uuid4().hex[:12]
        mock_env.reset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_session(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.core.session import SessionManager

        mock_registry = MagicMock()
        mock_env = AsyncMock()
        mock_env.reset = AsyncMock()
        mock_env.observe = AsyncMock()
        mock_registry.create.return_value = mock_env

        manager = SessionManager(registry=mock_registry)
        session = await manager.create_session("json", "test_task", sample_json_task_config)
        retrieved = manager.get_session(session.session_id)

        assert retrieved.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session_raises(self) -> None:
        from axiom.core.session import SessionManager
        from axiom.exceptions import SessionError

        mock_registry = MagicMock()
        manager = SessionManager(registry=mock_registry)

        with pytest.raises(SessionError):
            manager.get_session("nonexistent_id")

    @pytest.mark.asyncio
    async def test_close_session_calls_cleanup(self, sample_json_task_config: dict[str, Any]) -> None:
        from axiom.core.session import SessionManager

        mock_registry = MagicMock()
        mock_env = AsyncMock()
        mock_env.reset = AsyncMock()
        mock_env.observe = AsyncMock()
        mock_env.cleanup = AsyncMock()
        mock_registry.create.return_value = mock_env

        manager = SessionManager(registry=mock_registry)
        session = await manager.create_session("json", "test_task", sample_json_task_config)
        await manager.close_session(session.session_id)

        mock_env.cleanup.assert_awaited_once()

        from axiom.exceptions import SessionError
        with pytest.raises(SessionError):
            manager.get_session(session.session_id)


# ---------------------------------------------------------------------------
# Task loader tests
# ---------------------------------------------------------------------------


class TestTaskLoader:
    """TaskLoader must load and validate YAML task configs."""

    def test_load_task(self, tasks_dir: Path) -> None:
        from axiom.core.task_loader import TaskLoader

        loader = TaskLoader(tasks_dir=tasks_dir)
        config = loader.load_task("json", "test_task")

        assert config.name == "test_task"
        assert config.env == "json"
        assert config.max_steps == 5

    def test_load_nonexistent_task_raises(self, tasks_dir: Path) -> None:
        from axiom.core.task_loader import TaskLoader
        from axiom.exceptions import TaskConfigError

        loader = TaskLoader(tasks_dir=tasks_dir)
        with pytest.raises(TaskConfigError):
            loader.load_task("json", "nonexistent_task")

    def test_list_tasks(self, tasks_dir: Path) -> None:
        from axiom.core.task_loader import TaskLoader

        loader = TaskLoader(tasks_dir=tasks_dir)
        tasks = loader.list_tasks()

        assert len(tasks) >= 1
        assert any(t.name == "test_task" for t in tasks)

    def test_list_tasks_for_env(self, tasks_dir: Path) -> None:
        from axiom.core.task_loader import TaskLoader

        loader = TaskLoader(tasks_dir=tasks_dir)
        json_tasks = loader.list_tasks_for_env("json")

        assert all(t.env == "json" for t in json_tasks)


# ---------------------------------------------------------------------------
# Trajectory recorder tests
# ---------------------------------------------------------------------------


class TestTrajectoryRecorder:
    """TrajectoryRecorder must record steps and save trajectories."""

    def test_record_step(self) -> None:
        from axiom.core.trajectory import TrajectoryRecorder

        recorder = TrajectoryRecorder()
        recorder.start_session("sess_001", task_name="test_task", env_type="json")
        recorder.record_step(
            session_id="sess_001",
            step_num=1,
            action={"type": "api_call", "value": "add_todo", "params": {"title": "Buy milk"}},
            observation={"state": {"todos": [{"title": "Buy milk"}]}, "step_count": 1},
            reward=0.05,
            terminated=False,
            truncated=False,
        )

        trajectory = recorder.get_trajectory("sess_001")
        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].step_num == 1

    def test_save_trajectory(self, tmp_path: Path) -> None:
        from axiom.core.trajectory import TrajectoryRecorder

        recorder = TrajectoryRecorder()
        recorder.start_session("sess_002", task_name="test_task", env_type="json")
        recorder.record_step(
            session_id="sess_002",
            step_num=1,
            action={"type": "api_call", "value": "add_todo"},
            observation={"state": {"todos": []}},
            reward=0.05,
            terminated=False,
            truncated=False,
        )

        saved_path = recorder.save(session_id="sess_002", trajectory_dir=tmp_path)
        assert saved_path.exists()
        assert (saved_path / "trajectory.json").exists()

    def test_save_trajectory_with_screenshots(self, tmp_path: Path) -> None:
        """Screenshots must be saved as separate PNG files, not inline base64."""
        import base64

        from axiom.core.trajectory import TrajectoryRecorder

        # 1x1 red PNG pixel
        png_b64 = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode()

        recorder = TrajectoryRecorder()
        recorder.start_session("sess_003", task_name="test_task", env_type="webapp")
        recorder.record_step(
            session_id="sess_003",
            step_num=1,
            action={"type": "click", "selector": "#btn"},
            observation={"screenshot_base64": png_b64, "step_count": 1},
            reward=0.05,
            terminated=False,
            truncated=False,
        )

        saved_path = recorder.save(session_id="sess_003", trajectory_dir=tmp_path)
        screenshots_dir = saved_path / "screenshots"
        assert screenshots_dir.exists()
        assert (screenshots_dir / "step_1.png").exists()

        # Verify screenshot is NOT stored as base64 in the JSON
        import json

        trajectory_json = json.loads((saved_path / "trajectory.json").read_text())
        for step in trajectory_json["steps"]:
            assert "screenshot_base64" not in step.get("observation", {})
