"""Abstract base environment following the Gymnasium reset/step/observe interface.

This is the same pattern used by OSWorld, SWE-bench, and Deeptune's
training environments. Every concrete environment (JSON, WebApp, CLI)
inherits from BaseEnvironment.

Key design decisions:
  - Async context manager (__aenter__/__aexit__) guarantees cleanup
    even when exceptions occur. Critical at scale — Playwright browser
    leaks are a real production problem.
  - _ready flag prevents step()/observe() before reset().
  - Base class enforces max_steps truncation so subclasses don't duplicate it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from axiom.exceptions import EnvironmentNotReady

if TYPE_CHECKING:
    from types import TracebackType

    from axiom.models import Action, Observation, StepResult, TaskConfig


class BaseEnvironment(ABC):
    """Abstract base for all axiom environments.

    Subclasses must implement:
      - reset()   -> Observation
      - _step()   -> StepResult   (internal, called by public step())
      - observe() -> Observation
      - evaluate() -> dict[str, float]
      - cleanup()
      - env_id    (property)
    """

    def __init__(self, task_config: TaskConfig) -> None:
        self.task_config = task_config
        self.max_steps = task_config.max_steps
        self.step_count = 0
        self._ready = False

    # ------------------------------------------------------------------
    # Async context manager — guarantees cleanup
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BaseEnvironment:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.cleanup()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def reset(self) -> Observation:
        """Reset environment to initial state. Returns first observation.

        Subclasses MUST call super().reset() first.
        """
        self.step_count = 0
        self._ready = True
        return await self._reset()

    async def step(self, action: Action) -> StepResult:
        """Execute action and return result.

        Enforces:
          - EnvironmentNotReady if reset() not called
          - Max-steps truncation at the base class level
        """
        if not self._ready:
            msg = f"{self.__class__.__name__}: call reset() before step()"
            raise EnvironmentNotReady(msg)

        self.step_count += 1
        result = await self._step(action)

        # Enforce truncation at base class — subclasses don't need to check
        if self.step_count >= self.max_steps and not result.terminated:
            result.truncated = True

        return result

    async def observe(self) -> Observation:
        """Get current observation without acting."""
        if not self._ready:
            msg = f"{self.__class__.__name__}: call reset() before observe()"
            raise EnvironmentNotReady(msg)
        return await self._observe()

    # ------------------------------------------------------------------
    # Abstract methods — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def _reset(self) -> Observation:
        """Internal reset. Called by public reset() after bookkeeping."""
        ...

    @abstractmethod
    async def _step(self, action: Action) -> StepResult:
        """Internal step. Called by public step() after guards."""
        ...

    @abstractmethod
    async def _observe(self) -> Observation:
        """Internal observe. Called by public observe() after guards."""
        ...

    @abstractmethod
    async def evaluate(self) -> dict[str, float]:
        """Multi-signal evaluation.

        Returns dict with keys: completion, efficiency, accuracy, safety.
        All values in [0, 1].
        """
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources (close browser, remove temp files, etc).

        Must be idempotent — safe to call multiple times.
        """
        ...

    @property
    @abstractmethod
    def env_id(self) -> str:
        """Unique identifier for this environment type (e.g. 'json', 'webapp', 'cli')."""
        ...

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_truncated(self) -> bool:
        """True if step count has reached max_steps."""
        return self.step_count >= self.max_steps

    @property
    def metadata(self) -> dict[str, str | int]:
        """Environment metadata for logging and API responses."""
        return {
            "env_type": self.env_id,
            "task": self.task_config.name,
            "observation_mode": self.task_config.observation_mode,
            "max_steps": self.max_steps,
        }
