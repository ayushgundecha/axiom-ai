"""Base agent protocol.

Defines the interface that all agents must satisfy. Any class with
run_episode() and decide_action() methods works — no inheritance required.
"""

from __future__ import annotations

from typing import Any, Protocol


class BaseAgent(Protocol):
    """Protocol for axiom-ai agents.

    Agents interact with environments via the HTTP API, not directly.
    This decoupling means agents can be written in any language.
    """

    async def decide_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Choose the next action based on the current observation.

        Args:
            observation: Dict from the /observe or /step response.

        Returns:
            Action dict with keys: type, selector, value, params.
        """
        ...

    async def run_episode(
        self,
        base_url: str,
        env_name: str,
        task_id: str,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run a complete task episode via the HTTP API.

        Args:
            base_url: Axiom server URL (e.g. http://localhost:8000).
            env_name: Environment ID (json, webapp, cli).
            task_id: Task config ID.
            verbose: Print step-by-step progress.

        Returns:
            Evaluation scores dict.
        """
        ...
