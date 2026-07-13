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
  - Configurable model (defaults to claude-haiku-4-5-20251001)
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

import httpx

# Free-tier Gemini throttles aggressively (429) and flagship models shed load
# (503); a bounded backoff keeps a sequential sweep alive without masking real
# failures. Retryable codes only — auth/404 errors surface immediately.
GEMINI_RETRY_CODES = (429, 500, 503)
GEMINI_MAX_ATTEMPTS = 5


class ClaudeAgent:
    """AI agent powered by Claude that interacts with axiom environments."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.model = model
        # Route by model name so one agent serves both providers: Gemini
        # (free) for iteration, Claude for the on-brand frontier confirmation.
        # Clients are lazy so a Gemini-only run needs no Anthropic key (and vice
        # versa).
        self.provider = (
            "gemini"
            if any(k in model.lower() for k in ("gemini", "gemma"))
            else "anthropic"
        )
        self._anthropic: Any = None
        self._gemini: Any = None
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
            # AxiomChat's simplified DOM runs ~30-35k chars for a channel + open
            # thread; the reply input / send / resolve controls live near the END,
            # so a small window hides exactly the elements the agent must act on.
            dom = str(obs["dom_tree"])[:50000]
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
                "For browser actions, use a CSS selector that appears VERBATIM in the "
                "DOM above — copy the element's exact data-testid or href (e.g. "
                "[data-testid='dm-link-dm_u_lena'] or [data-testid='reply-send-m4']). "
                "Do NOT invent or guess selectors from display names/ids; prefer "
                "[data-testid='...']. To type, target the input's exact testid.\n\n"
                "Respond with ONLY ONE flat JSON object — no nesting, no markdown, no "
                "explanation. Use exactly one of these shapes:\n"
                '  {"type":"click","selector":"[data-testid=\'...\']"}\n'
                '  {"type":"type","selector":"[data-testid=\'...\']","value":"text to type"}\n'
                '  {"type":"press_key","value":"Enter"}\n'
                '  {"type":"done"}   (only when the task is fully complete)\n'
                'Do NOT wrap the action inside another object (no {"action": ...}).'
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
        text = self._complete(system, content_blocks)
        return self._parse_action(text)

    # ------------------------------------------------------------------
    # Provider-agnostic completion (Anthropic | Gemini)
    # ------------------------------------------------------------------

    def _complete(self, system: str, content_blocks: list[dict[str, Any]]) -> str:
        """Send one prompt to the configured provider; return the raw reply text."""
        if self.provider == "gemini":
            return self._complete_gemini(system, content_blocks)
        return self._complete_anthropic(system, content_blocks)

    def _complete_anthropic(self, system: str, content_blocks: list[dict[str, Any]]) -> str:
        import anthropic

        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic()
        response = self._anthropic.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": content_blocks}],
        )
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens
        return str(response.content[0].text).strip()

    def _complete_gemini(self, system: str, content_blocks: list[dict[str, Any]]) -> str:
        from google import genai
        from google.genai import types

        if self._gemini is None:
            self._gemini = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        parts: list[Any] = []
        for block in content_blocks:
            if block.get("type") == "image":
                parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(block["source"]["data"]),
                        mime_type="image/png",
                    )
                )
            elif block.get("type") == "text":
                parts.append(types.Part.from_text(text=str(block["text"])))
        from google.genai import errors

        response: Any = None
        for attempt in range(GEMINI_MAX_ATTEMPTS):
            try:
                response = self._gemini.models.generate_content(
                    model=self.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        system_instruction=system, max_output_tokens=512
                    ),
                )
                break
            except errors.APIError as exc:
                if exc.code not in GEMINI_RETRY_CODES or attempt == GEMINI_MAX_ATTEMPTS - 1:
                    raise
                time.sleep(2.0 * 2**attempt)
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self._total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
            self._total_output_tokens += getattr(usage, "candidates_token_count", 0) or 0
        return str(response.text or "").strip()

    def _parse_action(self, text: str) -> dict[str, Any]:
        """Parse Claude's response into a flat action dict.

        Robust to how models actually reply: markdown fences, a JSON object
        embedded in prose, and a nested ``{"action": {...}}`` /
        ``{"type":"action","action":{...}}`` wrapper (all seen from Sonnet). On
        genuine failure, returns a harmless no-op that gets a small penalty.
        """
        text = text.strip()
        # Strip markdown code fences if present.
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if "```" in text:
                text = text.rsplit("```", 1)[0]
            text = text.strip()

        obj: Any = None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)  # first {...} in prose
            if match:
                try:
                    obj = json.loads(match.group(0))
                except json.JSONDecodeError:
                    obj = None

        if not isinstance(obj, dict):
            return {"type": "press_key", "value": "Escape"}

        # Unwrap a nested action wrapper, e.g. {"action": {...}} or
        # {"type":"action","action":{...}}.
        inner = obj.get("action")
        if isinstance(inner, dict):
            obj = inner

        return obj
