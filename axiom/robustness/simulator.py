"""Offline AxiomChat simulator — a deterministic in-memory workspace + applier.

A faithful, dependency-free stand-in for a live AxiomChat: it builds an
oracle-state (the same shape ``GET /api/_oracle/state`` returns) for a seed, and
applies a ``list[Action]`` exactly the way ``apps/axiomchat/src/store.ts`` would
(reply-in-thread, post, resolve, react, pin, navigate, search). This lets the
robustness harness and the corpus run end-to-end — RRS table and all — with no
browser and no Docker, and keeps the deterministic tests fast.

Message ids are stable across seeds (``m_sq``/``m_inc``/``m_req``/``m_t1``…) so
the scenario locators are predictable; only the scenario *content* varies by
seed. Same (seed, action sequence) => byte-identical post-state.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from axiom.models import Action

MINUTE = 60_000
BASE_TS = 1_717_405_200_000  # 2024-06-03T09:00:00Z (matches seed.ts BASE_EPOCH)

USERS: list[dict[str, Any]] = [
    {"id": "u_me", "name": "Jordan Lee", "handle": "jordan", "role": "member", "status": "active"},
    {"id": "u_emma", "name": "Emma Wilson", "handle": "emma", "role": "owner", "status": "active"},
    {"id": "u_maya", "name": "Maya Chen", "handle": "maya", "role": "admin", "status": "active"},
    {
        "id": "u_diego",
        "name": "Diego Santos",
        "handle": "diego",
        "role": "member",
        "status": "away",
    },
    {
        "id": "u_priya",
        "name": "Priya Nair",
        "handle": "priya",
        "role": "member",
        "status": "active",
    },
    {"id": "u_sam", "name": "Sam Okafor", "handle": "sam", "role": "member", "status": "dnd"},
    {
        "id": "u_lena",
        "name": "Lena Petrova",
        "handle": "lena",
        "role": "member",
        "status": "active",
    },
    {
        "id": "u_aisha",
        "name": "Aisha Khan",
        "handle": "aisha",
        "role": "member",
        "status": "active",
    },
    {"id": "u_ravi", "name": "Ravi Mehta", "handle": "ravi", "role": "member", "status": "offline"},
    {
        "id": "u_carlos",
        "name": "Carlos Diaz",
        "handle": "carlos",
        "role": "guest",
        "status": "away",
    },
]

_ID_TO_HANDLE = {u["id"]: u["handle"] for u in USERS}

# --- Scenario variant pools (index 0 == the canonical seed-42 content) -------

_SUPPORT_POOL: list[dict[str, Any]] = [
    {
        "facts": ["audience", "4.2", "Okta", "entityId", "SAML"],
        "text": (
            "Enterprise customer says SSO via Okta broke after the 4.2 release — SAML "
            "responses now fail with `invalid_audience` (entityId mismatch). Fix?"
        ),
    },
    {
        "facts": ["signing secret", "401", "webhook", "rotate", "re-sync"],
        "text": (
            "Growth-plan customer's webhook deliveries stopped after they rotated the "
            "signing secret. Retries all 401. Correct way to re-sync?"
        ),
    },
    {
        "facts": ["504", "50k", "invoices", "export", "pagination"],
        "text": "A user can't export invoices over 50k rows — the download 504s. Workaround?",
    },
]

_INCIDENT_POOL: list[dict[str, Any]] = [
    {
        "severity": "SEV2",
        "facts": ["checkout-api", "5xx", "12%", "p99 3.2s", "09:14"],
        "text": (
            "🚨 Elevated 5xx on checkout-api starting ~09:14 UTC. Error rate ~12%, "
            "p99 3.2s. Investigating."
        ),
    },
    {
        "severity": "SEV1",
        "facts": ["auth-service", "0%", "14:02", "all regions", "login"],
        "text": (
            "🚨 SEV1: auth-service is down, login success rate dropped to 0% at 14:02 "
            "UTC. All regions affected."
        ),
    },
]

_REQUEST_POOL: list[dict[str, Any]] = [
    {
        "assignee": "u_diego",
        "text": (
            "Can someone own migrating the billing webhooks to the new event bus before "
            "the 4.3 cutover? It touches Stripe + the ledger service."
        ),
    },
    {
        "assignee": "u_ravi",
        "text": (
            "We need an owner for the incident postmortem automation — pulling timelines "
            "from #incidents into the wiki. Who has bandwidth?"
        ),
    },
]

_TRIAGE_ITEMS: list[dict[str, Any]] = [
    {
        "id": "m_t1",
        "severity": "SEV3",
        "assignee": "u_aisha",
        "text": "BUG: intermittent 500 on /api/invoices/export for tenants > 50k rows.",
    },
    {
        "id": "m_t2",
        "severity": "SEV2",
        "assignee": "u_ravi",
        "text": "BUG: password reset emails delayed 20+ min during peak.",
    },
    {
        "id": "m_t3",
        "severity": "SEV3",
        "assignee": "u_maya",
        "text": "BUG: search returns deleted threads for ~5 min (index lag).",
    },
]

# Canonical seed-42 constants (kept for tests that assert specific content).
SUPPORT_FACTS = _SUPPORT_POOL[0]["facts"]
INCIDENT_FACTS = _INCIDENT_POOL[0]["facts"]
INCIDENT_SEVERITY = _INCIDENT_POOL[0]["severity"]


def resolve_mentions(text: str, users: list[dict[str, Any]] | None = None) -> list[str]:
    """Resolve ``@handle`` tokens to userIds (mirrors store.ts extractMentions)."""
    table = {u["handle"].lower(): u["id"] for u in (users or USERS)}
    out: list[str] = []
    for match in re.finditer(r"@([a-z0-9_]+)", text, flags=re.IGNORECASE):
        uid = table.get(match.group(1).lower())
        if uid and uid not in out:
            out.append(uid)
    return out


def _msg(
    mid: str,
    channel_id: str,
    author_id: str,
    text: str,
    ts: int,
    *,
    thread_root_id: str | None = None,
    mentions: list[str] | None = None,
    resolved: bool = False,
    pinned: bool = False,
    reactions: dict[str, list[str]] | None = None,
    scenario: str | None = None,
    answer_facts: list[str] | None = None,
    severity: str | None = None,
    summary_facts: list[str] | None = None,
    correct_assignee_id: str | None = None,
    is_question: bool | None = None,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "id": mid,
        "channelId": channel_id,
        "authorId": author_id,
        "text": text,
        "ts": ts,
        "mentions": mentions if mentions is not None else resolve_mentions(text),
        "reactions": reactions or {},
        "pinned": pinned,
        "resolved": resolved,
    }
    if thread_root_id is not None:
        m["threadRootId"] = thread_root_id
    if scenario is not None:
        m["_scenario"] = scenario
    if answer_facts is not None:
        m["_answerFacts"] = answer_facts
    if severity is not None:
        m["_severity"] = severity
    if summary_facts is not None:
        m["_summaryFacts"] = summary_facts
    if correct_assignee_id is not None:
        m["_correctAssigneeId"] = correct_assignee_id
    if is_question is not None:
        m["_isQuestion"] = is_question
    return m


def _reply_count(messages: list[dict[str, Any]], root_id: str) -> int:
    return sum(1 for m in messages if m.get("threadRootId") == root_id)


def derive(messages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Recompute the ``derived`` ground-truth block (mirrors store.ts derive)."""
    return {
        "questions": [
            {
                "messageId": m["id"],
                "channelId": m["channelId"],
                "answerFacts": m.get("_answerFacts", []),
                "answered": _reply_count(messages, m["id"]) > 0,
                "resolved": m.get("resolved", False),
            }
            for m in messages
            if m.get("_scenario") == "support_question"
        ],
        "incidents": [
            {
                "messageId": m["id"],
                "channelId": m["channelId"],
                "severity": m.get("_severity"),
                "summaryFacts": m.get("_summaryFacts", []),
                "replyCount": _reply_count(messages, m["id"]),
            }
            for m in messages
            if m.get("_scenario") == "incident"
        ],
        "requests": [
            {
                "messageId": m["id"],
                "channelId": m["channelId"],
                "correctAssigneeId": m.get("_correctAssigneeId", ""),
                "correctAssigneeHandle": _ID_TO_HANDLE.get(m.get("_correctAssigneeId", ""), ""),
            }
            for m in messages
            if m.get("_scenario") == "dm_request"
        ],
        "triage": [
            {
                "messageId": m["id"],
                "severity": m.get("_severity"),
                "correctAssigneeId": m.get("_correctAssigneeId", ""),
            }
            for m in messages
            if m.get("_scenario") == "triage"
        ],
    }


def build_workspace(
    seed: int = 42,
    scale: str = "medium",
    *,
    include_triage: bool = True,
) -> dict[str, Any]:
    """Build a deterministic oracle-state for ``seed`` with all reward scenarios.

    Content varies by seed (which support/incident/request variant); ids are
    stable. For ``seed == 42`` the content matches the canonical fixtures.
    """
    sup = _SUPPORT_POOL[seed % len(_SUPPORT_POOL)]
    inc = _INCIDENT_POOL[seed % len(_INCIDENT_POOL)]
    req = _REQUEST_POOL[seed % len(_REQUEST_POOL)]

    messages: list[dict[str, Any]] = [
        _msg("m1", "c_support", "u_maya", "Anyone seen the support queue today?", BASE_TS),
        _msg(
            "m_sq",
            "c_support",
            "u_carlos",
            sup["text"],
            BASE_TS + 5 * MINUTE,
            scenario="support_question",
            is_question=True,
            answer_facts=sup["facts"],
        ),
        _msg(
            "m_inc",
            "c_incidents",
            "u_sam",
            inc["text"],
            BASE_TS + 10 * MINUTE,
            scenario="incident",
            severity=inc["severity"],
            summary_facts=inc["facts"],
        ),
        _msg(
            "m_ir1",
            "c_incidents",
            "u_maya",
            "Connection pool looks saturated.",
            BASE_TS + 12 * MINUTE,
            thread_root_id="m_inc",
        ),
        _msg(
            "m_ir2",
            "c_incidents",
            "u_ravi",
            "Rolling back the deploy now.",
            BASE_TS + 14 * MINUTE,
            thread_root_id="m_inc",
        ),
        _msg(
            "m_d1",
            "dm_u_lena",
            "u_lena",
            "hey, do you have 10 min later today?",
            BASE_TS + 20 * MINUTE,
        ),
        _msg(
            "m_req",
            "dm_u_lena",
            "u_lena",
            req["text"],
            BASE_TS + 25 * MINUTE,
            scenario="dm_request",
            is_question=True,
            correct_assignee_id=req["assignee"],
        ),
    ]

    if include_triage:
        for i, item in enumerate(_TRIAGE_ITEMS):
            messages.append(
                _msg(
                    item["id"],
                    "c_triage",
                    "u_maya",
                    item["text"],
                    BASE_TS + (40 + 2 * i) * MINUTE,
                    scenario="triage",
                    severity=item["severity"],
                    correct_assignee_id=item["assignee"],
                )
            )

    channels: list[dict[str, Any]] = [
        {
            "id": "c_support",
            "name": "support",
            "kind": "public",
            "topic": "",
            "memberIds": [],
            "lastReadTs": BASE_TS,
        },
        {
            "id": "c_incidents",
            "name": "incidents",
            "kind": "public",
            "topic": "",
            "memberIds": [],
            "lastReadTs": BASE_TS,
        },
        {
            "id": "dm_u_lena",
            "name": "Lena Petrova",
            "kind": "dm",
            "topic": "",
            "memberIds": ["u_me", "u_lena"],
            "lastReadTs": BASE_TS,
        },
    ]
    if include_triage:
        channels.append(
            {
                "id": "c_triage",
                "name": "triage",
                "kind": "private",
                "topic": "",
                "memberIds": [],
                "lastReadTs": BASE_TS,
            }
        )

    return {
        "id": "ws_axiom",
        "name": "Axiom Labs",
        "seed": seed,
        "scale": scale,
        "currentUserId": "u_me",
        "users": copy.deepcopy(USERS),
        "channels": channels,
        "messages": messages,
        "derived": derive(messages),
    }


# Backwards-compatible alias used widely by the tests.
def make_oracle_state(*, seed: int = 42, include_triage: bool = True) -> dict[str, Any]:
    return build_workspace(seed=seed, include_triage=include_triage)


def public_state(oracle_state: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``derived`` block and every ``_``-prefixed message field."""
    pub = copy.deepcopy(oracle_state)
    pub.pop("derived", None)
    for m in pub["messages"]:
        for key in list(m):
            if key.startswith("_"):
                del m[key]
    return pub


def with_agent_messages(
    state: dict[str, Any],
    new_messages: list[dict[str, Any]],
    *,
    resolve_target: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``state`` with ``new_messages`` appended + derived rebuilt."""
    post = copy.deepcopy(state)
    post["messages"].extend(copy.deepcopy(new_messages))
    if resolve_target is not None:
        for m in post["messages"]:
            if m["id"] == resolve_target:
                m["resolved"] = True
    post["derived"] = derive(post["messages"])
    return post


def agent_reply(
    mid: str,
    channel_id: str,
    text: str,
    *,
    thread_root_id: str | None = None,
    author_id: str = "u_me",
    mentions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a new agent-authored message (mentions auto-resolved from text)."""
    return _msg(
        mid,
        channel_id,
        author_id,
        text,
        BASE_TS + 1_000 * MINUTE,
        thread_root_id=thread_root_id,
        mentions=mentions if mentions is not None else resolve_mentions(text),
    )


# ---------------------------------------------------------------------------
# Action applier (faithful subset of store.ts mutations)
# ---------------------------------------------------------------------------

_TESTID_RE = re.compile(r"\[data-testid='([^']+)'\]")


def _testid(selector: str | None) -> str:
    if not selector:
        return ""
    m = _TESTID_RE.search(selector)
    return m.group(1) if m else ""


def _channel_of(messages: list[dict[str, Any]], message_id: str) -> str:
    for m in messages:
        if m["id"] == message_id:
            return str(m.get("channelId", ""))
    return ""


def apply_actions(
    state: dict[str, Any],
    actions: list[Action],
    current_user: str = "u_me",
) -> dict[str, Any]:
    """Apply a ``list[Action]`` to an oracle state and return the post-state."""
    post = copy.deepcopy(state)
    messages: list[dict[str, Any]] = post["messages"]
    typed: dict[str, str] = {}
    nav_channel = ""
    counter = len(messages) + 1

    def by_id(mid: str) -> dict[str, Any] | None:
        return next((m for m in messages if m["id"] == mid), None)

    for action in actions:
        tname = action.type.value
        testid = _testid(action.selector)

        if tname == "type" and action.value is not None:
            typed[testid] = action.value
            continue
        if tname != "click":
            continue

        if testid.startswith("channel-link-"):
            nav_channel = testid[len("channel-link-") :]
        elif testid.startswith("dm-link-"):
            nav_channel = testid[len("dm-link-") :]
        elif testid.startswith("reply-send-"):
            rid = testid[len("reply-send-") :]
            text = typed.get(f"reply-input-{rid}", "").strip()
            if text:
                messages.append(
                    _msg(
                        f"u{counter}",
                        _channel_of(messages, rid),
                        current_user,
                        text,
                        BASE_TS + counter * MINUTE,
                        thread_root_id=rid,
                    )
                )
                counter += 1
        elif testid == "send-button":
            text = typed.get("message-input", "").strip()
            if text and nav_channel:
                messages.append(
                    _msg(f"u{counter}", nav_channel, current_user, text, BASE_TS + counter * MINUTE)
                )
                counter += 1
        elif testid.startswith("resolve-thread-"):
            target = by_id(testid[len("resolve-thread-") :])
            if target is not None:
                target["resolved"] = True
        elif testid.startswith("react-"):
            target = by_id(testid[len("react-") :])
            if target is not None:
                reactions = target.setdefault("reactions", {})
                users = reactions.setdefault("✅", [])
                if current_user not in users:
                    users.append(current_user)
        elif testid.startswith("pin-"):
            target = by_id(testid[len("pin-") :])
            if target is not None:
                target["pinned"] = not target.get("pinned", False)
        # search-button / thread-open / others: no state change

    post["derived"] = derive(messages)
    return post
