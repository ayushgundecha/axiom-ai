"""AxiomChat environment — the deterministic mini-Slack, axiom's 4th env.

A thin subclass of :class:`WebAppEnvironment`. It customizes only two things:

  1. the reset BODY — passing a deterministic ``{seed, scale}`` to the app's
     ``POST /api/reset`` so every episode reproduces a known workspace, and
  2. the post-navigation readiness wait — blocking on the ``data-app-ready``
     marker the SPA sets after it hydrates, rather than relying on a timeout.

Everything else (observation, stepping, browser actions, goal evaluation,
cleanup) is inherited unchanged — the whole point of the WebAppEnvironment
abstraction.
"""

from __future__ import annotations

from typing import Any

from axiom.config import get_settings
from axiom.envs.webapp_env import WebAppEnvironment
from axiom.models import Action, TaskConfig

# The SPA sets this on its root element once it has hydrated from /api/state.
_READY_SELECTOR = '[data-app-ready="true"]'
# The store flips data-busy on <html> while a mutation+refetch is in flight;
# this selector matches once it has settled.
_IDLE_SELECTOR = "html:not([data-busy])"
_READY_TIMEOUT_MS = 10_000


class AxiomChatEnvironment(WebAppEnvironment):
    """Playwright environment for the AxiomChat Slack app (env_id="axiomchat")."""

    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__(task_config)
        # Default to the AxiomChat app URL when the task doesn't pin one
        # (the base WebAppEnvironment defaults to the todo app on :3000).
        if task_config.app_url is None:
            self._app_url = get_settings().axiomchat_app_url

    @property
    def env_id(self) -> str:
        return "axiomchat"

    async def _reset_server(self) -> dict[str, Any]:
        """Deterministic reset body. Falls back to a stable default workspace."""
        seed = self.task_config.seed if self.task_config.seed is not None else 1
        scale = self.task_config.scale or "medium"
        return {"seed": seed, "scale": scale}

    async def _wait_for_ready(self) -> None:
        """Wait on the app's readiness signal instead of a fixed timeout."""
        assert self._page is not None
        await self._page.wait_for_selector(_READY_SELECTOR, timeout=_READY_TIMEOUT_MS)

    async def _execute_browser_action(self, action: Action) -> None:
        """Run the action, then wait for the app to settle.

        AxiomChat mutates state with a POST and then refetches /api/state; the
        store flips ``data-busy`` on <html> for the duration. Blocking on
        ``html:not([data-busy])`` ensures the inherited per-step goal check
        observes the re-rendered DOM — a signal, not a timeout. Non-mutating
        actions (navigation, scroll) leave the flag unset, so this is a no-op.
        """
        await super()._execute_browser_action(action)
        assert self._page is not None
        await self._page.wait_for_selector(_IDLE_SELECTOR, timeout=_READY_TIMEOUT_MS)
