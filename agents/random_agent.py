"""Random baseline agent.

Takes random actions in any environment. Exists purely for comparison —
showing that Claude's scores are meaningfully better than random proves
the evaluation framework works and the agent is actually reasoning.
"""

from __future__ import annotations

import random
import re
from typing import Any

import httpx


class RandomAgent:
    """Agent that takes random actions. Baseline for comparison."""

    async def run_episode(
        self,
        base_url: str,
        env_name: str,
        task_id: str,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run a complete task episode with random actions."""
        async with httpx.AsyncClient(base_url=base_url, timeout=60) as http:
            resp = await http.post(
                "/sessions",
                json={"env_name": env_name, "task_id": task_id},
            )
            resp.raise_for_status()
            data = resp.json()
            session_id: str = data["session_id"]
            obs: dict[str, Any] = data["observation"]

            if verbose:
                print(f"\n  [Random Agent] Env: {env_name} | Task: {task_id}")

            terminated = False
            truncated = False
            total_reward = 0.0
            step = 0

            while not terminated and not truncated:
                step += 1
                action = self._random_action(obs)

                if verbose:
                    action_str = f"{action.get('type', '?')}"
                    if action.get("value"):
                        action_str += f" = '{str(action['value'])[:30]}'"
                    print(f"  Step {step}: {action_str}")

                resp = await http.post(
                    f"/sessions/{session_id}/step",
                    json=action,
                )
                resp.raise_for_status()
                result = resp.json()

                obs = result["observation"]
                reward = result["reward"]
                terminated = result["terminated"]
                truncated = result["truncated"]
                total_reward += reward

            # Evaluate
            resp = await http.post(f"/sessions/{session_id}/evaluate")
            resp.raise_for_status()
            evaluation: dict[str, Any] = resp.json()

            if verbose:
                scores = evaluation.get("scores", {})
                print(
                    f"  [Random] completion={scores.get('completion', 0):.1f} "
                    f"steps={step} reward={total_reward:.2f}\n"
                )

            await http.delete(f"/sessions/{session_id}")
            return evaluation

    def _random_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Generate a random action based on available action types."""
        available = obs.get("available_action_types", [])

        if "run_command" in available:
            return self._random_cli_action()
        elif "click" in available:
            return self._random_browser_action(obs)
        else:
            return self._random_api_action()

    def _random_cli_action(self) -> dict[str, Any]:
        """Random CLI command."""
        commands = ["ls", "pwd", "find . -type f", "cat readme.md", "mkdir test"]
        return {"type": "run_command", "value": random.choice(commands)}

    def _random_browser_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Random browser action, picking selectors from DOM if available."""
        selectors = self._extract_selectors(obs.get("dom_tree", ""))

        action_type = random.choice(["click", "type"])

        if action_type == "click" and selectors:
            return {"type": "click", "selector": random.choice(selectors)}
        elif action_type == "type" and selectors:
            words = ["hello", "test", "todo item", "buy milk", "fix bug"]
            input_selectors = [s for s in selectors if "input" in s.lower()] or selectors
            return {
                "type": "type",
                "selector": random.choice(input_selectors),
                "value": random.choice(words),
            }
        else:
            return {"type": "press_key", "value": "Tab"}

    def _random_api_action(self) -> dict[str, Any]:
        """Random JSON env action."""
        rand_title = f"Random-{random.randint(1, 100)}"
        ops = [
            {"type": "api_call", "value": "add_todo", "params": {"title": rand_title}},
            {"type": "api_call", "value": "add_todo", "params": {"title": "test"}},
        ]
        return random.choice(ops)

    def _extract_selectors(self, dom: str) -> list[str]:
        """Extract data-testid selectors from DOM tree."""
        pattern = r'data-testid="([^"]+)"'
        matches = re.findall(pattern, dom)
        return [f"[data-testid='{m}']" for m in matches] if matches else []
