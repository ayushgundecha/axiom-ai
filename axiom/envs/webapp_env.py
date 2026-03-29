"""WebApp environment — Playwright browser automation.

This is the centerpiece of axiom-ai, mapping directly to Deeptune's
OSWorld environments. A real web application runs (in Docker or locally)
and an AI agent interacts with it via Playwright browser automation.

How it works:
  1. A real app runs (todo app on port 3000)
  2. Agent observes via screenshots + simplified DOM tree
  3. Agent takes browser actions (click, type, press_key, select, scroll)
  4. Environment evaluates task completion by inspecting actual DOM state

Key design decisions:
  - Per-session BrowserContext (isolated cookies, storage, cache)
  - reset() calls /api/reset on the app to clear SERVER state too
  - wait_for_load_state("networkidle") instead of arbitrary timeouts
  - Action timeout of 5s — if selector doesn't appear, BrowserError
  - Partial accuracy scoring for incomplete tasks (RL reward shaping)
"""

from __future__ import annotations

import base64
import contextlib
from typing import Any

import httpx
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from axiom.core.base_env import BaseEnvironment
from axiom.exceptions import BrowserError
from axiom.logging import get_logger
from axiom.models import (
    Action,
    ActionType,
    Observation,
    ObservationMode,
    StepResult,
    TaskConfig,
)
from axiom.utils.dom_parser import extract_simplified_dom

logger = get_logger(__name__)

# Timeout for selector-based actions (milliseconds)
_ACTION_TIMEOUT_MS = 5000


class WebAppEnvironment(BaseEnvironment):
    """Playwright-powered browser environment.

    Mirrors how Deeptune/OSWorld environments work:
      - Real app runs in Docker or locally
      - Agent observes via screenshots + DOM / accessibility tree
      - Agent acts via click, type, press_key, select, scroll
      - Evaluation inspects actual DOM state (execution-based)
    """

    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__(task_config)
        self._app_url = task_config.app_url or "http://localhost:3000"
        self._observation_mode = ObservationMode(
            task_config.observation_mode or "hybrid"
        )
        self._playwright_ctx: Any = None  # playwright context manager
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._action_history: list[dict[str, Any]] = []
        self._screenshots: list[str] = []

    @property
    def env_id(self) -> str:
        return "webapp"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def _reset(self) -> Observation:
        self._action_history = []
        self._screenshots = []

        # Launch browser if not already running
        if self._browser is None:
            self._playwright_ctx = await async_playwright().start()
            self._browser = await self._playwright_ctx.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )

        # Close old context if exists (fresh cookies, storage per episode)
        if self._context is not None:
            await self._context.close()

        # Create fresh browser context — isolated per session
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        self._page = await self._context.new_page()

        # Reset SERVER state via API (critical — blueprint misses this)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self._app_url}/api/reset",
                    json={},
                )
        except httpx.HTTPError:
            logger.warning("webapp.reset_api_failed", app_url=self._app_url)

        # Navigate to the app
        await self._page.goto(self._app_url, wait_until="networkidle")

        # Run setup actions if specified in task config
        for setup in self.task_config.setup_actions:
            action = Action(**setup)
            await self._execute_browser_action(action)
            await self._page.wait_for_load_state("networkidle")

        return await self._observe()

    async def _step(self, action: Action) -> StepResult:
        assert self._page is not None  # guaranteed by _ready guard

        info: dict[str, Any] = {
            "action": action.model_dump(),
            "valid": True,
            "error": None,
        }
        reward = 0.0

        try:
            await self._execute_browser_action(action)
            # Wait for network to settle + DOM to re-render.
            # networkidle fires when no requests for 500ms, but JS
            # rendering happens AFTER the fetch callback completes.
            await self._page.wait_for_load_state("networkidle")
            await self._page.wait_for_timeout(150)
            reward = 0.05  # Small reward for valid action
        except Exception as e:
            info["valid"] = False
            info["error"] = str(e)
            reward = -0.1

        self._action_history.append({
            "step": self.step_count,
            "action": action.model_dump(),
            "reward": reward,
            "valid": info["valid"],
        })

        # Check if task goal is met
        goal_met = await self._check_goal()
        if goal_met:
            reward += 1.0

        observation = await self._observe()

        return StepResult(
            observation=observation,
            reward=reward,
            terminated=goal_met,
            truncated=False,  # Base class handles truncation
            info=info,
        )

    async def _observe(self) -> Observation:
        assert self._page is not None

        screenshot_b64: str | None = None
        dom_tree: str | None = None

        # Capture screenshot
        if self._observation_mode in (
            ObservationMode.SCREENSHOT,
            ObservationMode.HYBRID,
        ):
            screenshot_bytes = await self._page.screenshot(type="png")
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            self._screenshots.append(screenshot_b64)

        # Extract simplified DOM tree
        if self._observation_mode in (ObservationMode.DOM, ObservationMode.HYBRID):
            page_content = await self._page.content()
            dom_tree = extract_simplified_dom(page_content)

        return Observation(
            dom_tree=dom_tree,
            screenshot_base64=screenshot_b64,
            task_description=self.task_config.description,
            available_action_types=["click", "type", "press_key", "select", "scroll"],
            step_count=self.step_count,
            max_steps=self.max_steps,
            url=self._page.url,
        )

    async def evaluate(self) -> dict[str, float]:
        goal_met = await self._check_goal()
        optimal = self.task_config.optimal_steps or self.max_steps
        efficiency = (
            max(0.0, 1.0 - (self.step_count - optimal) / self.max_steps)
            if goal_met
            else 0.0
        )
        invalid = sum(1 for a in self._action_history if not a.get("valid", True))

        return {
            "completion": 1.0 if goal_met else 0.0,
            "efficiency": round(efficiency, 3),
            "accuracy": 1.0 if goal_met else await self._partial_accuracy(),
            "safety": round(max(0.0, 1.0 - (invalid * 0.15)), 3),
            "total_steps": self.step_count,
            "optimal_steps": optimal,
            "invalid_actions": invalid,
            "screenshots_captured": len(self._screenshots),
        }

    async def cleanup(self) -> None:
        """Release all Playwright resources. Idempotent."""
        await self._safe_close_context()
        await self._safe_close_browser()
        await self._safe_close_playwright()

    async def _safe_close_context(self) -> None:
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.close()
            self._context = None
            self._page = None

    async def _safe_close_browser(self) -> None:
        if self._browser is not None:
            with contextlib.suppress(Exception):
                await self._browser.close()
            self._browser = None

    async def _safe_close_playwright(self) -> None:
        if self._playwright_ctx is not None:
            with contextlib.suppress(Exception):
                await self._playwright_ctx.stop()
            self._playwright_ctx = None

    # ------------------------------------------------------------------
    # Browser action execution
    # ------------------------------------------------------------------

    async def _execute_browser_action(self, action: Action) -> None:
        """Execute a browser action via Playwright."""
        assert self._page is not None

        if action.type == ActionType.CLICK:
            if action.selector:
                await self._page.click(
                    action.selector, timeout=_ACTION_TIMEOUT_MS
                )
            elif action.params and "x" in action.params and "y" in action.params:
                # Coordinate-based click
                await self._page.mouse.click(
                    float(action.params["x"]),
                    float(action.params["y"]),
                )
            else:
                msg = "CLICK requires a selector or x/y coordinates"
                raise BrowserError(msg)

        elif action.type == ActionType.TYPE:
            if action.selector and action.value is not None:
                await self._page.fill(
                    action.selector,
                    action.value,
                    timeout=_ACTION_TIMEOUT_MS,
                )
            elif action.value is not None:
                await self._page.keyboard.type(action.value)
            else:
                msg = "TYPE requires a value"
                raise BrowserError(msg)

        elif action.type == ActionType.PRESS_KEY:
            key = action.value or "Enter"
            await self._page.keyboard.press(key)

        elif action.type == ActionType.SELECT:
            if action.selector and action.value is not None:
                await self._page.select_option(
                    action.selector,
                    action.value,
                    timeout=_ACTION_TIMEOUT_MS,
                )
            else:
                msg = "SELECT requires selector and value"
                raise BrowserError(msg)

        elif action.type == ActionType.SCROLL:
            direction = action.value or "down"
            delta = 300 if direction == "down" else -300
            await self._page.mouse.wheel(0, delta)

        else:
            msg = f"Unsupported browser action: {action.type}"
            raise BrowserError(msg)

    # ------------------------------------------------------------------
    # Goal checking — inspects actual DOM state
    # ------------------------------------------------------------------

    async def _check_goal(self) -> bool:
        """Check task completion by inspecting the actual DOM.

        This is execution-based evaluation — we check the REAL state
        of the app, not a simulated one.
        """
        assert self._page is not None
        goal = self.task_config.goal
        goal_type = goal.get("type")

        if goal_type == "elements_exist":
            selectors: list[str] = goal.get("selectors", [])
            for sel in selectors:
                element = await self._page.query_selector(sel)
                if not element:
                    return False
            return True

        elif goal_type == "text_content_matches":
            checks: list[dict[str, str]] = goal.get("checks", [])
            for check in checks:
                selector = check["selector"]
                expected = check["contains"]
                element = await self._page.query_selector(selector)
                if not element:
                    return False
                text = await element.text_content()
                if expected not in (text or ""):
                    return False
            return True

        elif goal_type == "element_count":
            selector = str(goal.get("selector", ""))
            expected_count = int(goal.get("count", 0))
            elements = await self._page.query_selector_all(selector)
            return bool(len(elements) == expected_count)

        elif goal_type == "custom_js":
            js_code = str(goal.get("script", "return false;"))
            result = await self._page.evaluate(js_code)
            return bool(result)

        return False

    async def _partial_accuracy(self) -> float:
        """Partial credit for incomplete tasks."""
        assert self._page is not None
        goal = self.task_config.goal
        if goal.get("type") == "element_count":
            selector = str(goal.get("selector", ""))
            expected = int(goal.get("count", 0))
            actual = len(await self._page.query_selector_all(selector))
            if expected > 0:
                return float(round(min(actual, expected) / expected, 3))
        return 0.0
