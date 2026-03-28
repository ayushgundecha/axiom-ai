"""Tests for axiom/models.py — Pydantic data models.

These tests define the contract for all core data types.
Written TDD-style before implementation.
"""

import pytest
from pydantic import ValidationError


class TestActionType:
    """ActionType enum must define all supported action types."""

    def test_browser_action_types_exist(self) -> None:
        from axiom.models import ActionType

        assert ActionType.CLICK == "click"
        assert ActionType.TYPE == "type"
        assert ActionType.PRESS_KEY == "press_key"
        assert ActionType.SELECT == "select"
        assert ActionType.SCROLL == "scroll"

    def test_cli_action_type_exists(self) -> None:
        from axiom.models import ActionType

        assert ActionType.RUN_COMMAND == "run_command"

    def test_api_action_type_exists(self) -> None:
        from axiom.models import ActionType

        assert ActionType.API_CALL == "api_call"


class TestObservationMode:
    """ObservationMode enum must define all observation modes."""

    def test_all_modes_exist(self) -> None:
        from axiom.models import ObservationMode

        assert ObservationMode.JSON == "json"
        assert ObservationMode.DOM == "dom"
        assert ObservationMode.SCREENSHOT == "screenshot"
        assert ObservationMode.HYBRID == "hybrid"
        assert ObservationMode.TEXT == "text"


class TestAction:
    """Action model with validation rules."""

    def test_valid_click_action(self) -> None:
        from axiom.models import Action, ActionType

        action = Action(type=ActionType.CLICK, selector="#my-button")
        assert action.type == ActionType.CLICK
        assert action.selector == "#my-button"

    def test_click_without_selector_raises(self) -> None:
        from axiom.models import Action, ActionType

        with pytest.raises(ValidationError):
            Action(type=ActionType.CLICK)

    def test_valid_type_action(self) -> None:
        from axiom.models import Action, ActionType

        action = Action(type=ActionType.TYPE, selector="#input", value="hello")
        assert action.value == "hello"

    def test_type_without_value_raises(self) -> None:
        from axiom.models import Action, ActionType

        with pytest.raises(ValidationError):
            Action(type=ActionType.TYPE, selector="#input")

    def test_valid_run_command_action(self) -> None:
        from axiom.models import Action, ActionType

        action = Action(type=ActionType.RUN_COMMAND, value="ls -la")
        assert action.value == "ls -la"

    def test_run_command_without_value_raises(self) -> None:
        from axiom.models import Action, ActionType

        with pytest.raises(ValidationError):
            Action(type=ActionType.RUN_COMMAND)

    def test_valid_api_call_action(self) -> None:
        from axiom.models import Action, ActionType

        action = Action(
            type=ActionType.API_CALL,
            value="add_todo",
            params={"title": "Buy milk"},
        )
        assert action.params == {"title": "Buy milk"}

    def test_press_key_action(self) -> None:
        from axiom.models import Action, ActionType

        action = Action(type=ActionType.PRESS_KEY, value="Enter")
        assert action.value == "Enter"

    def test_scroll_action_no_value_ok(self) -> None:
        from axiom.models import Action, ActionType

        # Scroll can work without value (defaults to down)
        action = Action(type=ActionType.SCROLL)
        assert action.type == ActionType.SCROLL


class TestObservation:
    """Observation model must carry environment state for agents."""

    def test_minimal_observation(self) -> None:
        from axiom.models import Observation

        obs = Observation(
            task_description="Do something",
            available_action_types=["click", "type"],
            step_count=0,
            max_steps=10,
        )
        assert obs.task_description == "Do something"
        assert obs.dom_tree is None
        assert obs.screenshot_base64 is None

    def test_full_observation(self) -> None:
        from axiom.models import Observation

        obs = Observation(
            dom_tree="<div>hello</div>",
            screenshot_base64="iVBORw...",
            text_output=None,
            state={"todos": []},
            task_description="Add todos",
            available_action_types=["click", "type"],
            step_count=3,
            max_steps=10,
            url="http://localhost:3000",
        )
        assert obs.dom_tree is not None
        assert obs.step_count == 3


class TestStepResult:
    """StepResult returned after every environment step."""

    def test_step_result_fields(self) -> None:
        from axiom.models import Observation, StepResult

        obs = Observation(
            task_description="test",
            available_action_types=["click"],
            step_count=1,
            max_steps=10,
        )
        result = StepResult(
            observation=obs,
            reward=0.05,
            terminated=False,
            truncated=False,
            info={"valid": True},
        )
        assert result.reward == 0.05
        assert not result.terminated
        assert result.info["valid"] is True


class TestEvaluationResult:
    """EvaluationResult must carry all four scoring dimensions."""

    def test_evaluation_result_fields(self) -> None:
        from axiom.models import EvaluationResult

        result = EvaluationResult(
            completion=1.0,
            efficiency=0.85,
            accuracy=1.0,
            safety=1.0,
            total_steps=6,
            optimal_steps=6,
            invalid_actions=0,
        )
        assert result.completion == 1.0
        assert result.efficiency == 0.85

    def test_scores_must_be_in_range(self) -> None:
        from axiom.models import EvaluationResult

        # Scores should be 0-1 range
        with pytest.raises(ValidationError):
            EvaluationResult(
                completion=1.5,  # Out of range
                efficiency=0.5,
                accuracy=0.5,
                safety=0.5,
                total_steps=1,
                optimal_steps=1,
                invalid_actions=0,
            )


class TestTaskConfig:
    """TaskConfig parsed from YAML task definitions."""

    def test_parse_minimal_config(self) -> None:
        from axiom.models import TaskConfig

        config = TaskConfig(
            name="test_task",
            env="json",
            description="A test task.",
            max_steps=10,
            goal={"type": "element_count", "field": "todos", "count": 2},
        )
        assert config.name == "test_task"
        assert config.env == "json"
        assert config.max_steps == 10

    def test_parse_webapp_config_with_setup_actions(self) -> None:
        from axiom.models import TaskConfig

        config = TaskConfig(
            name="complete_todos",
            env="webapp",
            description="Complete all todos.",
            app_url="http://localhost:3000",
            observation_mode="hybrid",
            max_steps=10,
            optimal_steps=3,
            setup_actions=[
                {"type": "type", "selector": "#input", "value": "Task 1"},
                {"type": "click", "selector": "#add-btn"},
            ],
            goal={"type": "custom_js", "script": "return true;"},
        )
        assert len(config.setup_actions) == 2
        assert config.app_url == "http://localhost:3000"

    def test_parse_cli_config_with_initial_state(self) -> None:
        from axiom.models import TaskConfig

        config = TaskConfig(
            name="organize_files",
            env="cli",
            description="Organize files by type.",
            max_steps=10,
            initial_state={
                "files": [
                    {"path": "photo.png", "content": ""},
                    {"path": "readme.md", "content": "# Hello"},
                ]
            },
            goal={
                "type": "directory_structure",
                "expected_structure": ["images/photo.png", "docs/readme.md"],
            },
        )
        assert len(config.initial_state["files"]) == 2
