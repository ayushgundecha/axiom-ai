"""Environment registry — pluggable environment discovery and creation.

A simple dict-based registry with a decorator for clean registration.
When a concrete environment module is imported, the @register_decorator
fires and adds the class to the registry.

Design: intentionally simple. No metaclasses, no plugin discovery,
no dynamic imports. A dict with a decorator is all we need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from axiom.exceptions import TaskConfigError

if TYPE_CHECKING:
    from axiom.core.base_env import BaseEnvironment
    from axiom.models import TaskConfig


class EnvironmentRegistry:
    """Registry for environment classes.

    Usage:
        registry = EnvironmentRegistry()

        @registry.register_decorator("json")
        class JSONEnvironment(BaseEnvironment):
            ...

        env = registry.create("json", task_config)
    """

    def __init__(self) -> None:
        self._envs: dict[str, type[Any]] = {}

    def register(self, env_id: str, env_class: type[Any]) -> None:
        """Register an environment class by ID."""
        self._envs[env_id] = env_class

    def register_decorator(self, env_id: str) -> Any:
        """Class decorator that registers an environment on import.

        Usage:
            @registry.register_decorator("webapp")
            class WebAppEnvironment(BaseEnvironment):
                ...
        """

        def decorator(cls: type[Any]) -> type[Any]:
            self.register(env_id, cls)
            return cls

        return decorator

    def create(self, env_id: str, task_config: TaskConfig) -> BaseEnvironment:
        """Create an environment instance by ID.

        Raises TaskConfigError if the env_id is not registered.
        """
        env_class = self._envs.get(env_id)
        if env_class is None:
            available = ", ".join(sorted(self._envs)) or "(none)"
            msg = f"Unknown environment '{env_id}'. Available: {available}"
            raise TaskConfigError(msg)
        return env_class(task_config)  # type: ignore[no-any-return]

    def list_envs(self) -> list[str]:
        """Return sorted list of registered environment IDs."""
        return sorted(self._envs)

    def get_env_class(self, env_id: str) -> type[Any]:
        """Return the class for a registered environment ID.

        Raises TaskConfigError if not found.
        """
        env_class = self._envs.get(env_id)
        if env_class is None:
            msg = f"Unknown environment '{env_id}'"
            raise TaskConfigError(msg)
        return env_class
