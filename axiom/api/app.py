"""FastAPI application factory.

Uses the application factory pattern: create_app() returns a configured
FastAPI instance. State (registry, session manager, etc.) is initialized
eagerly in create_app() so it's available immediately. The lifespan
async context manager handles graceful shutdown (close all sessions).

This split is intentional: sync setup in the factory, async teardown
in the lifespan. Works correctly with both uvicorn and TestClient.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from axiom.api.middleware import RequestIdMiddleware, register_exception_handlers
from axiom.api.routes import environments, health, sessions, tasks
from axiom.config import get_settings
from axiom.core.registry import EnvironmentRegistry
from axiom.core.session import SessionManager
from axiom.core.task_loader import TaskLoader
from axiom.core.trajectory import TrajectoryRecorder
from axiom.logging import configure_logging, get_logger


def _create_registry() -> EnvironmentRegistry:
    """Create and populate the environment registry."""
    from axiom.envs.json_env import JSONEnvironment

    registry = EnvironmentRegistry()
    registry.register("json", JSONEnvironment)
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: handles graceful shutdown."""
    logger = get_logger(__name__)
    logger.info(
        "server.started",
        envs=app.state.registry.list_envs(),
    )

    yield

    # Shutdown: close all active sessions
    await app.state.session_manager.close_all()
    logger.info("server.stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    All state is initialized eagerly here (not in lifespan) so it's
    available immediately — works with both uvicorn and TestClient.
    """
    settings = get_settings()
    configure_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
    )

    app = FastAPI(
        title="axiom-ai",
        description="Training gym for AI agents operating in real digital environments",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Initialize core components on app.state
    registry = _create_registry()
    app.state.registry = registry
    app.state.session_manager = SessionManager(
        registry=registry,
        max_age_seconds=settings.max_session_age_seconds,
    )
    app.state.task_loader = TaskLoader(tasks_dir=Path("tasks"))
    app.state.trajectory_recorder = TrajectoryRecorder()

    # Middleware
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    # Routes
    app.include_router(health.router)
    app.include_router(environments.router)
    app.include_router(tasks.router)
    app.include_router(sessions.router)

    return app
