"""Token-gated client for the AxiomChat privileged oracle endpoint.

The robustness harness — and ONLY the harness — fetches the privileged
``GET /api/_oracle/state`` view, which exposes the hidden ``_``-prefixed
ground-truth labels and the structured ``derived`` block. Access is gated by
the ``X-Oracle-Token`` header (config ``axiomchat_oracle_token``). The token
lives here and in the harness; it is never handed to an agent or environment.

The ``derived`` block (produced by ``apps/axiomchat/src/store.ts``) has shape::

    {
      "questions":  [{messageId, channelId, answerFacts[], answered, resolved}],
      "incidents":  [{messageId, channelId, severity, summaryFacts[], replyCount}],
      "requests":   [{messageId, channelId, correctAssigneeId, correctAssigneeHandle}],
      "triage":     [{messageId, severity, correctAssigneeId}],
    }

This module is pure I/O + parsing; the deterministic grading lives in
:mod:`axiom.robustness.oracles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from axiom.config import get_settings
from axiom.exceptions import OracleError

# Maps a scenario name (the hidden ``_scenario`` label, or a friendlier alias)
# to the key under ``derived`` that holds its ground truth.
_SCENARIO_TO_DERIVED_KEY: dict[str, str] = {
    "support_question": "questions",
    "questions": "questions",
    "support": "questions",
    "incident": "incidents",
    "incidents": "incidents",
    "dm_request": "requests",
    "requests": "requests",
    "request": "requests",
    "triage": "triage",
}


async def fetch_oracle_state(
    base_url: str,
    token: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    request_timeout: float = 10.0,
) -> dict[str, Any]:
    """Fetch the privileged oracle state from a running AxiomChat.

    Performs ``GET {base_url}/api/_oracle/state`` with the ``X-Oracle-Token``
    header and returns the parsed JSON, including the ``derived`` ground-truth
    block. When ``token`` is omitted the configured
    ``get_settings().axiomchat_oracle_token`` is used.

    Args:
        base_url: AxiomChat origin, e.g. ``http://localhost:3100``.
        token: Oracle token; defaults to the configured one.
        client: Optional pre-built async client (for testing / connection reuse).
        request_timeout: Request timeout in seconds (ignored when ``client`` given).

    Raises:
        OracleError: on 403 (missing/invalid token), any other non-200 status,
            a transport failure, or a payload missing the ``derived`` block.
    """
    tok = token if token is not None else get_settings().axiomchat_oracle_token
    url = f"{base_url.rstrip('/')}/api/_oracle/state"
    headers = {"X-Oracle-Token": tok}

    try:
        if client is not None:
            resp = await client.get(url, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=request_timeout) as owned:
                resp = await owned.get(url, headers=headers)
    except httpx.HTTPError as exc:  # transport-level failure
        msg = f"could not reach oracle endpoint {url}: {exc}"
        raise OracleError(msg) from exc

    if resp.status_code == 403:
        msg = (
            f"oracle endpoint {url} returned 403 — a valid X-Oracle-Token is "
            "required (the token is harness-side only)"
        )
        raise OracleError(msg)
    if resp.status_code != 200:
        msg = f"oracle endpoint {url} returned status {resp.status_code}"
        raise OracleError(msg)

    try:
        data: dict[str, Any] = resp.json()
    except ValueError as exc:
        msg = f"oracle endpoint {url} returned non-JSON body"
        raise OracleError(msg) from exc

    if "derived" not in data:
        msg = "oracle state is missing the 'derived' ground-truth block"
        raise OracleError(msg)
    return data


def derived_for_scenario(oracle_state: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    """Return the ``derived`` ground-truth items for a scenario.

    Accepts either a hidden ``_scenario`` label (``support_question``,
    ``incident``, ``dm_request``, ``triage``) or the ``derived`` key itself
    (``questions``, ``incidents``, ``requests``, ``triage``). Returns an empty
    list when the scenario is absent for this seed (e.g. ``triage`` is optional).
    """
    derived = oracle_state.get("derived", {})
    if not isinstance(derived, dict):
        return []
    key = _SCENARIO_TO_DERIVED_KEY.get(scenario, scenario)
    items = derived.get(key, [])
    return list(items) if isinstance(items, list) else []


# ---------------------------------------------------------------------------
# Agent artifacts — the riskiest piece (RR2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentArtifact:
    """One message the agent (``current_user``) authored during an episode.

    Produced by diffing a pre-episode oracle/public state against the
    post-episode one. Carries exactly what every deterministic grader needs —
    the reply ``text`` and resolved ``mentions`` (userIds) — mapped back to its
    thread/channel so it can be matched against the right ``derived`` target.
    """

    message_id: str
    channel_id: str
    thread_root_id: str | None
    text: str
    mentions: tuple[str, ...]


def message_ids(state: dict[str, Any]) -> set[str]:
    """Return the set of message ids present in a workspace state."""
    return {
        m["id"]
        for m in state.get("messages", [])
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    }


def agent_artifacts(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    current_user: str = "u_me",
    root_id: str | None = None,
    channel_id: str | None = None,
) -> list[AgentArtifact]:
    """Diff two states and return the NEW messages authored by ``current_user``.

    This is pure data diffing against the workspace ``messages`` array — no DOM
    scraping, no LLM — so it is fully deterministic and unit-testable. It works
    on either the public (``/api/state``) or privileged (``/api/_oracle/state``)
    state, since both carry ``id``/``authorId``/``threadRootId``/``channelId``/
    ``text``/``mentions`` on every message.

    Args:
        pre_state: workspace state snapshotted at episode reset.
        post_state: workspace state after the episode.
        current_user: the userId the agent acts as (AxiomChat: ``u_me``).
        root_id: if set, keep only replies whose ``threadRootId`` equals it.
        channel_id: if set, keep only messages in this channel.

    Returns:
        New ``current_user``-authored messages in ``post_state`` order
        (post_state is itself deterministic per seed + action sequence).
    """
    seen = message_ids(pre_state)
    out: list[AgentArtifact] = []
    for m in post_state.get("messages", []):
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not isinstance(mid, str) or mid in seen:
            continue
        if m.get("authorId") != current_user:
            continue
        if root_id is not None and m.get("threadRootId") != root_id:
            continue
        if channel_id is not None and m.get("channelId") != channel_id:
            continue
        raw_mentions = m.get("mentions") or []
        mentions = tuple(x for x in raw_mentions if isinstance(x, str))
        out.append(
            AgentArtifact(
                message_id=mid,
                channel_id=str(m.get("channelId", "")),
                thread_root_id=m.get("threadRootId"),
                text=str(m.get("text", "")),
                mentions=mentions,
            )
        )
    return out


def artifact_texts(artifacts: list[AgentArtifact]) -> list[str]:
    """Reply texts of the agent's artifacts, in order."""
    return [a.text for a in artifacts]


def artifact_mentions(artifacts: list[AgentArtifact]) -> list[list[str]]:
    """Per-artifact mention lists (parallel to :func:`artifact_texts`)."""
    return [list(a.mentions) for a in artifacts]


def mention_set(artifacts: list[AgentArtifact]) -> set[str]:
    """Union of all userIds mentioned across the agent's artifacts."""
    out: set[str] = set()
    for a in artifacts:
        out.update(a.mentions)
    return out
