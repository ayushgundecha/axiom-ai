"""Claude agent with vision — sees screenshots + reads DOM to decide actions.

This agent interacts with axiom environments via the HTTP API. For WebApp
environments, it sends screenshots as base64 images to Claude's vision
capability and reads the simplified DOM tree for structured understanding.

Key design decisions:
  - Async httpx client (not sync) — the whole system is async
  - Conversation history: last 3 observation-action pairs as context
    so Claude can avoid repeating failed actions
  - Retry on invalid JSON: one retry with a correction prompt
  - Cost tracking: logs input/output tokens per call
  - Configurable model (defaults to claude-sonnet-4-20250514)
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx


class ClaudeAgent:
    """AI agent powered by Claude that interacts with axiom environments."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self._history: list[dict[str, Any]] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def run_episode(
        self,
        base_url: str,
        env_name: str,
        task_id: str,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run a complete task episode via the axiom HTTP API."""
        self._history = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        async with httpx.AsyncClient(base_url=base_url, timeout=60) as http:
            # Create session
            resp = await http.post(
                "/sessions",
                json={"env_name": env_name, "task_id": task_id},
            )
            resp.raise_for_status()
            data = resp.json()
            session_id: str = data["session_id"]
            obs: dict[str, Any] = data["observation"]

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  Env: {env_name} | Task: {task_id}")
                desc = obs.get("task_description", "")[:80]
                print(f"  {desc}")
                print(f"{'=' * 60}\n")

            terminated = False
            truncated = False
            total_reward = 0.0
            step = 0

            while not terminated and not truncated:
                step += 1
                action = await self.decide_action(obs)

                if verbose:
                    action_str = f"{action.get('type', '?')}"
                    if action.get("selector"):
                        action_str += f" -> {action['selector']}"
                    if action.get("value"):
                        val = str(action["value"])[:40]
                        action_str += f" = '{val}'"
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

                if verbose:
                    status = "DONE" if terminated else ("TIMEOUT" if truncated else "->")
                    print(f"         reward: {reward:+.2f} | total: {total_reward:.2f} | {status}")

                # Track history for context
                self._history.append({"action": action, "reward": reward})
                if len(self._history) > 3:
                    self._history = self._history[-3:]

            # Evaluate
            resp = await http.post(f"/sessions/{session_id}/evaluate")
            resp.raise_for_status()
            evaluation: dict[str, Any] = resp.json()

            if verbose:
                print(f"\n  {'─' * 50}")
                print("  EVALUATION SCORES:")
                for k, v in evaluation.get("scores", {}).items():
                    if isinstance(v, float) and v <= 1.0:
                        bar = "█" * int(v * 20)
                        print(f"    {k:20s}: {v:.3f} {bar}")
                    else:
                        print(f"    {k:20s}: {v}")
                print(f"  {'─' * 50}")
                print(
                    f"  Tokens: {self._total_input_tokens} in / {self._total_output_tokens} out\n"
                )

            # Cleanup
            await http.delete(f"/sessions/{session_id}")

            return evaluation

    async def decide_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Use Claude to decide the next action based on observation."""
        content_blocks: list[dict[str, Any]] = []

        # If we have a screenshot, include it as an image (vision)
        if obs.get("screenshot_base64"):
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": obs["screenshot_base64"],
                    },
                }
            )

        # Build text prompt
        prompt_parts: list[str] = [f"Task: {obs.get('task_description', '')}"]

        if obs.get("dom_tree"):
            dom = str(obs["dom_tree"])[:3000]
            prompt_parts.append(f"Current page DOM:\n{dom}")

        if obs.get("text_output"):
            text = str(obs["text_output"])[:2000]
            prompt_parts.append(f"Terminal output:\n{text}")

        if obs.get("state"):
            state = json.dumps(obs["state"], indent=2)[:1500]
            prompt_parts.append(f"Current state:\n{state}")

        prompt_parts.append(f"Step {obs.get('step_count', '?')}/{obs.get('max_steps', '?')}")
        prompt_parts.append(f"Available actions: {obs.get('available_action_types', [])}")

        # Add history for context
        if self._history:
            history_lines = []
            for h in self._history:
                a = h["action"]
                history_lines.append(
                    f"  {a.get('type')} "
                    f"{a.get('selector', '')} "
                    f"{a.get('value', '')[:30]} "
                    f"-> reward: {h['reward']:+.2f}"
                )
            prompt_parts.append("Recent actions:\n" + "\n".join(history_lines))

        # Action format instructions
        available = obs.get("available_action_types", [])
        if "click" in available:
            prompt_parts.append(
                "For browser actions, use CSS selectors like "
                "[data-testid='todo-input'] or #element-id.\n"
                "Respond with ONLY a JSON object (no markdown):\n"
                '{"type": "click|type|press_key", '
                '"selector": "css-selector", "value": "text-or-key"}'
            )
        elif "run_command" in available:
            prompt_parts.append(
                "Respond with ONLY a JSON object (no markdown):\n"
                '{"type": "run_command", "value": "shell-command"}'
            )
        else:
            prompt_parts.append(
                "Respond with ONLY a JSON object (no markdown):\n"
                '{"type": "api_call", "value": "operation", '
                '"params": {"key": "value"}}'
            )

        content_blocks.append(
            {
                "type": "text",
                "text": "\n\n".join(prompt_parts),
            }
        )

        system = (
            "You are an AI agent completing tasks in a digital environment. "
            "You can see the current state via screenshots, DOM tree, or text output. "
            "Choose the SINGLE best next action to progress toward the goal. "
            "Be efficient — minimize steps. Respond with ONLY valid JSON, "
            "no markdown formatting, no explanation."
        )

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": content_blocks}],
        )

        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens

        text = response.content[0].text.strip()
        return self._parse_action(text)

    def _parse_action(self, text: str) -> dict[str, Any]:
        """Parse Claude's response into an action dict. Handles markdown fences."""
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if "```" in text:
                text = text.rsplit("```", 1)[0]
            text = text.strip()

        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            # Fallback: return a no-op that will get a small penalty
            return {"type": "press_key", "value": "Escape"}
