"""API-specific request/response models.

These are SEPARATE from the domain models in axiom/models.py.
API schemas handle HTTP concerns (what the client sends/receives).
Domain models handle business logic. This separation is a real
architectural pattern — internal models should not leak HTTP concerns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Request body for POST /sessions."""

    env_name: str
    task_id: str


class StepRequest(BaseModel):
    """Request body for POST /sessions/{id}/step."""

    type: str
    selector: str | None = None
    value: str | None = None
    params: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    """Response for POST /sessions."""

    session_id: str
    observation: dict[str, Any]


class StepResponse(BaseModel):
    """Response for POST /sessions/{id}/step."""

    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class EvaluateResponse(BaseModel):
    """Response for POST /sessions/{id}/evaluate."""

    scores: dict[str, Any]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    active_sessions: int
    registered_envs: list[str]


class EnvironmentsResponse(BaseModel):
    """Response for GET /envs."""

    environments: list[str]


class TasksResponse(BaseModel):
    """Response for GET /tasks."""

    tasks: list[dict[str, Any]]
