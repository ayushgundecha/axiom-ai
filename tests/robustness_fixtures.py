"""Robustness test fixtures — thin re-exports of the package simulator.

The deterministic offline AxiomChat (workspace builder + action applier) now
lives in :mod:`axiom.robustness.simulator` so the harness can reuse it. This
module just re-exports the names the test suite already imports.
"""

from __future__ import annotations

from axiom.robustness.simulator import (
    BASE_TS,
    INCIDENT_FACTS,
    INCIDENT_SEVERITY,
    MINUTE,
    SUPPORT_FACTS,
    USERS,
    agent_reply,
    apply_actions,
    build_workspace,
    derive,
    make_oracle_state,
    public_state,
    resolve_mentions,
    with_agent_messages,
)

__all__ = [
    "BASE_TS",
    "INCIDENT_FACTS",
    "INCIDENT_SEVERITY",
    "MINUTE",
    "SUPPORT_FACTS",
    "USERS",
    "agent_reply",
    "apply_actions",
    "build_workspace",
    "derive",
    "make_oracle_state",
    "public_state",
    "resolve_mentions",
    "with_agent_messages",
]
