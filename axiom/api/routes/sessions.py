"""Session CRUD and step/observe/evaluate endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from axiom.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    EvaluateResponse,
    StepRequest,
    StepResponse,
)
from axiom.models import Action, ActionType

router = APIRouter(prefix="/sessions")


@router.post("", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    request: Request,
) -> CreateSessionResponse:
    """Create a new environment session."""
    state = request.app.state
    task_config = state.task_loader.load_task(body.env_name, body.task_id)

    session = await state.session_manager.create_session(
        env_name=body.env_name,
        task_id=body.task_id,
        task_config=task_config.model_dump(),
    )

    # Record trajectory start
    state.trajectory_recorder.start_session(
        session_id=session.session_id,
        task_name=task_config.name,
        env_type=body.env_name,
    )

    obs = await session.env.observe()
    return CreateSessionResponse(
        session_id=session.session_id,
        observation=obs.model_dump(),
    )


@router.post("/{session_id}/step", response_model=StepResponse)
async def step_session(
    session_id: str,
    body: StepRequest,
    request: Request,
) -> StepResponse:
    """Take an action in an environment session."""
    state = request.app.state
    session = state.session_manager.get_session(session_id)

    action = Action(
        type=ActionType(body.type),
        selector=body.selector,
        value=body.value,
        params=body.params,
    )

    result = await session.env.step(action)

    # Record step in trajectory
    state.trajectory_recorder.record_step(
        session_id=session_id,
        step_num=session.env.step_count,
        action=action.model_dump(),
        observation=result.observation.model_dump(),
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
    )

    return StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
        info=result.info,
    )


@router.get("/{session_id}/observe")
async def observe_session(
    session_id: str,
    request: Request,
) -> dict[str, Any]:
    """Get current observation without acting."""
    session = request.app.state.session_manager.get_session(session_id)
    obs = await session.env.observe()
    result: dict[str, Any] = obs.model_dump()
    return result


@router.post("/{session_id}/evaluate", response_model=EvaluateResponse)
async def evaluate_session(
    session_id: str,
    request: Request,
) -> EvaluateResponse:
    """Run multi-signal evaluation on the current environment state."""
    session = request.app.state.session_manager.get_session(session_id)
    scores = await session.env.evaluate()
    return EvaluateResponse(scores=scores)


@router.get("/{session_id}/trajectory")
async def get_trajectory(
    session_id: str,
    request: Request,
) -> dict[str, Any]:
    """Return the recorded trajectory for a session."""
    # Verify session exists (raises SessionError -> 404 if not)
    request.app.state.session_manager.get_session(session_id)
    trajectory = request.app.state.trajectory_recorder.get_trajectory(session_id)
    result: dict[str, Any] = trajectory.model_dump()
    return result


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    request: Request,
) -> None:
    """Close and cleanup a session."""
    await request.app.state.session_manager.close_session(session_id)
