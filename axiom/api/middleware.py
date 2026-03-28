"""API middleware: request ID injection and exception handling.

Request ID middleware generates a UUID per request and binds it to
structlog context — every log line for that request includes request_id.

Exception handlers catch AxiomError subtypes and return appropriate
HTTP status codes with structured JSON error responses.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from axiom.exceptions import (
    AxiomError,
    EnvironmentNotReady,
    SessionError,
    TaskConfigError,
)
from axiom.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers that map AxiomError subtypes to HTTP status codes."""

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "session_not_found", "detail": str(exc)},
        )

    @app.exception_handler(TaskConfigError)
    async def task_config_error_handler(request: Request, exc: TaskConfigError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_task_config", "detail": str(exc)},
        )

    @app.exception_handler(EnvironmentNotReady)
    async def env_not_ready_handler(request: Request, exc: EnvironmentNotReady) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "environment_not_ready", "detail": str(exc)},
        )

    @app.exception_handler(AxiomError)
    async def axiom_error_handler(request: Request, exc: AxiomError) -> JSONResponse:
        logger.error("unhandled_axiom_error", error=str(exc), error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(exc)},
        )


class RequestIdMiddleware:
    """ASGI middleware that adds a unique request_id to every request.

    Binds request_id to structlog context so all log lines within
    a request include it automatically.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            import structlog

            request_id = uuid.uuid4().hex[:12]
            structlog.contextvars.bind_contextvars(request_id=request_id)

            async def send_with_id(message: Any) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", request_id.encode()))
                    message["headers"] = headers
                await send(message)

            try:
                await self.app(scope, receive, send_with_id)
            finally:
                structlog.contextvars.unbind_contextvars("request_id")
        else:
            await self.app(scope, receive, send)
