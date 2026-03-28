"""Task listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from axiom.api.schemas import TasksResponse

router = APIRouter()


@router.get("/tasks", response_model=TasksResponse)
async def list_tasks(request: Request) -> TasksResponse:
    """Return list of available task configurations."""
    task_loader = request.app.state.task_loader
    tasks = task_loader.list_tasks()
    return TasksResponse(
        tasks=[t.model_dump() for t in tasks],
    )
