"""Shared pytest fixtures for axiom-ai test suite."""

import asyncio
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Async event loop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Sample task configs (used across multiple test modules)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_json_task_config() -> dict[str, Any]:
    """Minimal JSON environment task config for testing."""
    return {
        "name": "test_add_todos",
        "env": "json",
        "description": "Add two todos to the list.",
        "observation_mode": "json",
        "max_steps": 10,
        "optimal_steps": 2,
        "goal": {
            "type": "element_count",
            "field": "todos",
            "count": 2,
        },
    }


@pytest.fixture
def sample_webapp_task_config() -> dict[str, Any]:
    """Minimal WebApp environment task config for testing."""
    return {
        "name": "test_add_three_todos",
        "env": "webapp",
        "description": "Add three todos to the todo app.",
        "app_url": os.getenv("TODO_APP_URL", "http://localhost:3000"),
        "observation_mode": "hybrid",
        "max_steps": 12,
        "optimal_steps": 6,
        "goal": {
            "type": "element_count",
            "selector": "[data-testid^='todo-item-']",
            "count": 3,
        },
    }


@pytest.fixture
def sample_cli_task_config() -> dict[str, Any]:
    """Minimal CLI environment task config for testing."""
    return {
        "name": "test_organize_files",
        "env": "cli",
        "description": "Create a docs/ directory and move readme.md into it.",
        "observation_mode": "text",
        "max_steps": 10,
        "optimal_steps": 2,
        "initial_state": {
            "files": [
                {"path": "readme.md", "content": "# Test Project"},
                {"path": "notes.txt", "content": "Some notes"},
            ]
        },
        "goal": {
            "type": "directory_structure",
            "expected_structure": ["docs/readme.md"],
        },
    }


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    """Create a temporary tasks directory with sample YAML files."""
    json_dir = tmp_path / "tasks" / "json"
    json_dir.mkdir(parents=True)

    task_yaml = json_dir / "test_task.yaml"
    task_yaml.write_text(
        "name: test_task\n"
        "env: json\n"
        "description: A test task.\n"
        "observation_mode: json\n"
        "max_steps: 5\n"
        "optimal_steps: 2\n"
        "goal:\n"
        "  type: element_count\n"
        "  field: todos\n"
        "  count: 1\n"
    )
    return tmp_path / "tasks"


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


def is_todo_app_running() -> bool:
    """Check if the todo web app is accessible."""
    import httpx

    url = os.getenv("TODO_APP_URL", "http://localhost:3000")
    try:
        resp = httpx.get(f"{url}/api/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# Skip marker for integration tests requiring the todo app
requires_todo_app = pytest.mark.skipif(
    not is_todo_app_running(),
    reason="Todo app not running (start with: docker-compose up todo-app)",
)
