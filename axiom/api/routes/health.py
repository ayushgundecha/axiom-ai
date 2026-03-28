"""Health check endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from axiom.api.schemas import HealthResponse

if TYPE_CHECKING:
    pass

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return server status, active session count, and registered environments."""
    state = request.app.state
    return HealthResponse(
        status="ok",
        active_sessions=state.session_manager.active_count,
        registered_envs=state.registry.list_envs(),
    )
