"""Structured logging configuration using structlog.

Two output modes controlled by AXIOM_LOG_FORMAT:
  - "console" (dev): colorful, human-readable output
  - "json" (production): JSON lines for log aggregation

Every log line from within a session automatically includes session_id
via structlog's context variables.

Usage:
    from axiom.logging import configure_logging, get_logger
    configure_logging(log_level="info", log_format="console")
    logger = get_logger()
    logger.info("server.started", port=8000)
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "info", log_format: str = "console") -> None:
    """Configure structlog for the application.

    Args:
        log_level: Python log level name (debug, info, warning, error).
        log_format: "console" for dev (colorful) or "json" for production.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger instance.

    Args:
        name: Logger name. Defaults to caller's module name.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_session_context(session_id: str) -> None:
    """Bind session_id to structlog context for all subsequent log calls.

    Call this when a session is created so every log line includes
    the session ID automatically.
    """
    structlog.contextvars.bind_contextvars(session_id=session_id)


def clear_session_context() -> None:
    """Clear session context after a request completes."""
    structlog.contextvars.unbind_contextvars("session_id")
