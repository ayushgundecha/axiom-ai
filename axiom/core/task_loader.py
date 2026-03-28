"""YAML task configuration loader.

Loads task definitions from the tasks/ directory, validates them against
the TaskConfig Pydantic model, and provides lookup by env type.

Directory structure:
    tasks/
        json/
            create_and_complete.yaml
        webapp/
            add_three_todos.yaml
        cli/
            organize_files.yaml
"""

from __future__ import annotations

from pathlib import Path

import yaml

from axiom.exceptions import TaskConfigError
from axiom.models import TaskConfig


class TaskLoader:
    """Loads and validates YAML task configurations.

    Args:
        tasks_dir: Root directory containing env-type subdirectories.
    """

    def __init__(self, tasks_dir: Path) -> None:
        self._tasks_dir = tasks_dir

    def load_task(self, env_type: str, task_id: str) -> TaskConfig:
        """Load a specific task by env type and task ID.

        Looks for: tasks_dir/{env_type}/{task_id}.yaml

        Raises TaskConfigError if file not found or validation fails.
        """
        task_file = self._tasks_dir / env_type / f"{task_id}.yaml"

        if not task_file.exists():
            msg = f"Task file not found: {task_file}. Expected at tasks/{env_type}/{task_id}.yaml"
            raise TaskConfigError(msg)

        try:
            raw = yaml.safe_load(task_file.read_text())
            return TaskConfig(**raw)
        except Exception as e:
            msg = f"Invalid task config in {task_file}: {e}"
            raise TaskConfigError(msg) from e

    def list_tasks(self) -> list[TaskConfig]:
        """Load and return all task configs from all env-type directories."""
        tasks: list[TaskConfig] = []

        if not self._tasks_dir.exists():
            return tasks

        for env_dir in sorted(self._tasks_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            for task_file in sorted(env_dir.glob("*.yaml")):
                try:
                    raw = yaml.safe_load(task_file.read_text())
                    tasks.append(TaskConfig(**raw))
                except Exception:
                    # Skip malformed files during listing
                    continue

        return tasks

    def list_tasks_for_env(self, env_id: str) -> list[TaskConfig]:
        """Load and return all task configs for a specific environment type."""
        env_dir = self._tasks_dir / env_id

        if not env_dir.exists():
            return []

        tasks: list[TaskConfig] = []
        for task_file in sorted(env_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(task_file.read_text())
                tasks.append(TaskConfig(**raw))
            except Exception:
                continue

        return tasks
