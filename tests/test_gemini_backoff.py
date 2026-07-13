"""Gemini provider routing + rate-limit backoff (free-tier resilience, no API needed)."""

from __future__ import annotations

from typing import Any

from google.genai import errors

from agents.claude_agent import ClaudeAgent


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "synthetic"}})


class _FlakyModels:
    """generate_content raises `fail_codes` in order, then succeeds."""

    def __init__(self, fail_codes: list[int]) -> None:
        self.fail_codes = list(fail_codes)
        self.calls = 0

    def generate_content(self, **_: Any) -> Any:
        self.calls += 1
        if self.fail_codes:
            raise _api_error(self.fail_codes.pop(0))

        class _Resp:
            text = '{"type": "click", "selector": "[data-testid=\'send-button\']"}'
            usage_metadata = None

        return _Resp()


class _FakeGeminiClient:
    def __init__(self, fail_codes: list[int]) -> None:
        self.models = _FlakyModels(fail_codes)


def test_gemma_models_route_to_gemini_provider() -> None:
    assert ClaudeAgent(model="gemma-4-31b-it").provider == "gemini"
    assert ClaudeAgent(model="gemini-3.1-flash-lite").provider == "gemini"
    assert ClaudeAgent(model="claude-haiku-4-5-20251001").provider == "anthropic"


def test_gemini_retries_on_429_then_succeeds(monkeypatch: Any) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    agent = ClaudeAgent(model="gemini-3.1-flash-lite")
    agent._gemini = _FakeGeminiClient(fail_codes=[429, 503])
    text = agent._complete_gemini("system", [{"type": "text", "text": "hi"}])
    assert "send-button" in text
    assert agent._gemini.models.calls == 3


def test_gemini_does_not_retry_non_transient_errors(monkeypatch: Any) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    agent = ClaudeAgent(model="gemini-3.1-flash-lite")
    agent._gemini = _FakeGeminiClient(fail_codes=[404])
    try:
        agent._complete_gemini("system", [{"type": "text", "text": "hi"}])
    except errors.APIError as exc:
        assert exc.code == 404
        assert agent._gemini.models.calls == 1
    else:
        raise AssertionError("expected APIError(404) to propagate")
