"""Custom exception hierarchy for axiom-ai.

Every exception inherits from AxiomError so callers can catch broadly
or narrowly. The API layer maps these to HTTP status codes:
  - SessionError        -> 404
  - TaskConfigError     -> 422
  - EnvironmentError    -> 500
  - EvaluationError     -> 500
"""


class AxiomError(Exception):
    """Base exception for all axiom-ai errors."""


# ---------------------------------------------------------------------------
# Environment errors
# ---------------------------------------------------------------------------


class EnvironmentError(AxiomError):  # noqa: A001  — intentional shadow of builtin
    """Base for errors occurring within an environment."""


class BrowserError(EnvironmentError):
    """Playwright / browser automation failure."""


class CommandError(EnvironmentError):
    """CLI subprocess execution failure."""


class EnvironmentNotReady(EnvironmentError):
    """Raised when step() or observe() is called before reset()."""


# ---------------------------------------------------------------------------
# Configuration & task errors
# ---------------------------------------------------------------------------


class TaskConfigError(AxiomError):
    """Invalid or missing task configuration."""


# ---------------------------------------------------------------------------
# Session errors
# ---------------------------------------------------------------------------


class SessionError(AxiomError):
    """Session not found, expired, or otherwise invalid."""


# ---------------------------------------------------------------------------
# Evaluation errors
# ---------------------------------------------------------------------------


class EvaluationError(AxiomError):
    """Failure during goal checking or score computation."""


# ---------------------------------------------------------------------------
# Robustness / oracle errors (Pillar 2)
# ---------------------------------------------------------------------------


class OracleError(AxiomError):
    """Failure fetching or parsing the privileged oracle state.

    Raised by the robustness harness when the token-gated
    ``GET /api/_oracle/state`` endpoint is unreachable, returns a non-200
    status (e.g. 403 when the ``X-Oracle-Token`` is missing or wrong), or
    returns a payload missing the ``derived`` ground-truth block.
    """
