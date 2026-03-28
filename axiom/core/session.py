"""Session lifecycle manager.

Manages active environment sessions: creation, lookup, cleanup, and
expiration. Each session wraps an environment instance and tracks
metadata (creation time, last activity).

Thread safety: uses asyncio.Lock for session dict mutations since
multiple concurrent API requests may touch sessions simultaneously.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from axiom.exceptions import SessionError
from axiom.logging import get_logger

if TYPE_CHECKING:
    from axiom.core.base_env import BaseEnvironment
    from axiom.core.registry import EnvironmentRegistry

logger = get_logger(__name__)


@dataclass
class Session:
    """An active environment session."""

    session_id: str
    env: BaseEnvironment
    task_id: str
    env_name: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = time.time()


class SessionManager:
    """Manages active environment sessions.

    Args:
        registry: EnvironmentRegistry for creating environments.
        max_age_seconds: Auto-expire sessions older than this.
    """

    def __init__(
        self,
        registry: EnvironmentRegistry,
        max_age_seconds: int = 3600,
    ) -> None:
        self._registry = registry
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._max_age_seconds = max_age_seconds

    async def create_session(
        self,
        env_name: str,
        task_id: str,
        task_config: dict[str, Any],
    ) -> Session:
        """Create a new session: instantiate env, call reset().

        Returns a Session with a short, URL-friendly ID.
        """
        from axiom.models import TaskConfig

        config = TaskConfig(**task_config)
        env = self._registry.create(env_name, config)

        session_id = uuid.uuid4().hex[:12]
        await env.reset()

        session = Session(
            session_id=session_id,
            env=env,
            task_id=task_id,
            env_name=env_name,
        )

        async with self._lock:
            self._sessions[session_id] = session

        logger.info(
            "session.created",
            session_id=session_id,
            env_name=env_name,
            task_id=task_id,
        )
        return session

    def get_session(self, session_id: str) -> Session:
        """Retrieve an active session by ID.

        Raises SessionError if not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            msg = f"Session '{session_id}' not found"
            raise SessionError(msg)
        session.touch()
        return session

    async def close_session(self, session_id: str) -> None:
        """Close a session: call env.cleanup(), remove from active sessions."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)

        if session is None:
            msg = f"Session '{session_id}' not found"
            raise SessionError(msg)

        await session.env.cleanup()
        logger.info("session.closed", session_id=session_id)

    async def close_all(self) -> None:
        """Close all active sessions. Called on server shutdown."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            try:
                await session.env.cleanup()
            except Exception:
                logger.exception(
                    "session.cleanup_failed",
                    session_id=session.session_id,
                )

    async def close_expired(self) -> int:
        """Close sessions older than max_age_seconds. Returns count closed."""
        now = time.time()
        expired_ids: list[str] = []

        async with self._lock:
            for sid, session in self._sessions.items():
                if now - session.last_active > self._max_age_seconds:
                    expired_ids.append(sid)

        import contextlib

        for sid in expired_ids:
            with contextlib.suppress(SessionError):
                await self.close_session(sid)

        return len(expired_ids)

    @property
    def active_count(self) -> int:
        """Number of currently active sessions."""
        return len(self._sessions)
