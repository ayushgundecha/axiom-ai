"""Unit tests for the token-gated oracle client (RR1)."""

from __future__ import annotations

import httpx
import pytest

from axiom.exceptions import OracleError
from axiom.robustness.oracle_client import derived_for_scenario, fetch_oracle_state
from tests.robustness_fixtures import make_oracle_state

BASE_URL = "http://axiomchat.test:3100"
GOOD_TOKEN = "axiom-oracle-dev-token"


def _handler(state: dict, *, expected_token: str = GOOD_TOKEN):
    """Build a MockTransport handler mimicking GET /api/_oracle/state."""

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/_oracle/state"
        token = request.headers.get("X-Oracle-Token")
        if token != expected_token:
            return httpx.Response(403, json={"error": "forbidden"})
        return httpx.Response(200, json=state)

    return handle


async def test_fetch_returns_derived_block_for_valid_token():
    state = make_oracle_state(seed=7)
    transport = httpx.MockTransport(_handler(state))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await fetch_oracle_state(BASE_URL, GOOD_TOKEN, client=client)

    assert result["seed"] == 7
    assert "derived" in result
    assert {"questions", "incidents", "requests", "triage"} <= set(result["derived"])
    assert result["derived"]["questions"][0]["answerFacts"]


async def test_fetch_raises_oracle_error_on_403():
    transport = httpx.MockTransport(_handler(make_oracle_state()))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OracleError, match="403"):
            await fetch_oracle_state(BASE_URL, "wrong-token", client=client)


async def test_fetch_raises_on_missing_token():
    transport = httpx.MockTransport(_handler(make_oracle_state()))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OracleError, match="403"):
            await fetch_oracle_state(BASE_URL, "", client=client)


async def test_fetch_raises_on_non200():
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OracleError, match="status 500"):
            await fetch_oracle_state(BASE_URL, GOOD_TOKEN, client=client)


async def test_fetch_raises_when_derived_missing():
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "ws", "messages": []})

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OracleError, match="derived"):
            await fetch_oracle_state(BASE_URL, GOOD_TOKEN, client=client)


async def test_fetch_raises_on_transport_error():
    def handle(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(OracleError, match="could not reach"):
            await fetch_oracle_state(BASE_URL, GOOD_TOKEN, client=client)


def test_derived_for_scenario_accepts_scenario_label_and_key():
    state = make_oracle_state()
    by_label = derived_for_scenario(state, "support_question")
    by_key = derived_for_scenario(state, "questions")
    assert by_label == by_key
    assert by_label[0]["messageId"] == "m_sq"

    assert derived_for_scenario(state, "incident")[0]["severity"] == "SEV2"
    assert derived_for_scenario(state, "dm_request")[0]["correctAssigneeId"] == "u_diego"
    assert len(derived_for_scenario(state, "triage")) == 3


def test_derived_for_scenario_empty_when_absent():
    state = make_oracle_state(include_triage=False)
    assert derived_for_scenario(state, "triage") == []
    assert derived_for_scenario(state, "nonsense") == []
