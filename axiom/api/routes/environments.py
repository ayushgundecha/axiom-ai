"""Environment listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from axiom.api.schemas import EnvironmentsResponse

router = APIRouter()


@router.get("/envs", response_model=EnvironmentsResponse)
async def list_environments(request: Request) -> EnvironmentsResponse:
    """Return list of registered environment IDs."""
    return EnvironmentsResponse(
        environments=request.app.state.registry.list_envs(),
    )
