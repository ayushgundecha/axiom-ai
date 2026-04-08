"""Application configuration via pydantic-settings.

Uses environment variables with the AXIOM_ prefix. Loads from .env file
automatically. Every configurable value lives here — no hardcoded strings
elsewhere in the codebase.

Usage:
    from axiom.config import get_settings
    settings = get_settings()
    print(settings.todo_app_url)
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AxiomSettings(BaseSettings):
    """Typed configuration for axiom-ai.

    All fields can be overridden via environment variables prefixed with AXIOM_.
    Example: AXIOM_LOG_LEVEL=debug overrides log_level.
    """

    model_config = SettingsConfigDict(
        env_prefix="AXIOM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"  # noqa: S104 — bind all interfaces intentionally
    port: int = 8000

    # Todo app (target application for WebApp environment)
    todo_app_url: str = "http://localhost:3000"

    # Logging
    log_level: str = "info"
    log_format: str = "console"  # "console" (dev) or "json" (production)

    # Storage
    screenshot_dir: Path = Path("./trajectories")
    trajectory_dir: Path = Path("./trajectories")

    # Sessions
    max_session_age_seconds: int = 3600

    # LLM Judge
    llm_judge_model: str = "claude-haiku-4-5-20251001"
    llm_judge_enabled: bool = False


@lru_cache(maxsize=1)
def get_settings() -> AxiomSettings:
    """Return cached settings instance. Call once at startup."""
    return AxiomSettings()
