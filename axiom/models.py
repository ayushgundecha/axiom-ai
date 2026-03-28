"""Pydantic data models for axiom-ai.

This module defines every data type the system uses — the vocabulary layer.
All other modules import from here. Models use Pydantic v2 with strict
validation and custom validators where the blueprint used raw dicts.

Key design decisions:
  - Action has a model_validator enforcing that click needs a selector,
    type needs a value, run_command needs a value, etc.
  - EvaluationResult scores are constrained to [0, 1].
  - TaskConfig is strongly typed (not a raw dict) so YAML misconfigs
    are caught at load time, not at runtime.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ObservationMode(StrEnum):
    """How the agent observes the environment.

    Mirrors real agent architectures:
      - JSON: structured state (simple envs)
      - DOM: DOM / accessibility tree (computer-use)
      - SCREENSHOT: base64 PNG (vision agents)
      - HYBRID: DOM + screenshot (recommended for LLMs)
      - TEXT: plain text output (CLI environments)
    """

    JSON = "json"
    DOM = "dom"
    SCREENSHOT = "screenshot"
    HYBRID = "hybrid"
    TEXT = "text"


class ActionType(StrEnum):
    """Types of actions an agent can take.

    Browser actions map to Playwright operations.
    CLI actions map to subprocess execution.
    API actions map to direct function calls (JSON env).
    """

    # Browser actions (computer-use)
    CLICK = "click"
    TYPE = "type"
    PRESS_KEY = "press_key"
    SELECT = "select"
    SCROLL = "scroll"

    # CLI actions
    RUN_COMMAND = "run_command"

    # JSON env actions
    API_CALL = "api_call"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """An action the agent wants to take.

    Validation rules enforce that each action type has the fields it needs:
      - CLICK requires selector
      - TYPE requires value
      - RUN_COMMAND requires value
    """

    type: ActionType
    selector: str | None = None
    value: str | None = None
    params: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> Action:
        """Ensure required fields are present for each action type."""
        if self.type == ActionType.CLICK and not self.selector:
            msg = "CLICK action requires a selector"
            raise ValueError(msg)
        if self.type == ActionType.TYPE and not self.value:
            msg = "TYPE action requires a value"
            raise ValueError(msg)
        if self.type == ActionType.RUN_COMMAND and not self.value:
            msg = "RUN_COMMAND action requires a value (the command to run)"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """What the agent sees after each step.

    Different environments populate different fields:
      - WebApp: dom_tree + screenshot_base64 + url
      - CLI: text_output
      - JSON: state
    All share: task_description, available_action_types, step/max counts.
    """

    dom_tree: str | None = None
    screenshot_base64: str | None = None
    text_output: str | None = None
    state: dict[str, Any] | None = None
    task_description: str
    available_action_types: list[str]
    step_count: int
    max_steps: int
    url: str | None = None


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """Returned after every environment step.

    Follows the Gymnasium convention:
      - observation: what the agent sees now
      - reward: immediate scalar reward
      - terminated: task goal was met
      - truncated: hit max steps without completing
      - info: extra metadata (action validity, errors, etc.)
    """

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Multi-signal evaluation scores.

    All four core scores are constrained to [0, 1]:
      - completion: did the agent finish the task?
      - efficiency: steps taken vs optimal
      - accuracy: how correct is the final state?
      - safety: did it avoid destructive / invalid actions?
    """

    completion: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    total_steps: int = Field(ge=0)
    optimal_steps: int = Field(ge=0)
    invalid_actions: int = Field(ge=0)


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------


class TaskConfig(BaseModel):
    """Strongly-typed task configuration parsed from YAML.

    The blueprint used raw dicts everywhere for task_config. This model
    catches misconfigured YAML at load time instead of at runtime.
    """

    name: str
    env: str
    description: str
    max_steps: int = 20
    optimal_steps: int | None = None
    observation_mode: str = "hybrid"
    app_url: str | None = None
    goal: dict[str, Any]
    setup_actions: list[dict[str, Any]] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TrajectoryStep(BaseModel):
    """A single step in a recorded trajectory."""

    step_num: int
    action: dict[str, Any]
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool


class Trajectory(BaseModel):
    """Full recorded trajectory for one episode."""

    session_id: str
    task_name: str
    env_type: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
