"""JSON state-machine environment — the simplest concrete environment.

A pure Python environment with in-memory dict state. No Docker, no
browser, no subprocess — just a dict and some operations. Exists to:
  1. Validate the entire core framework end-to-end
  2. Provide a cheap baseline for testing agents
  3. Demonstrate the Gymnasium reset/step/observe interface

State: {"todos": [{id, title, completed}, ...]}
Operations: add_todo, complete_todo, delete_todo (via API_CALL actions)
Goal checking: inspects the state dict directly
"""

from __future__ import annotations

import uuid
from typing import Any

from axiom.core.base_env import BaseEnvironment
from axiom.models import (
    Action,
    ActionType,
    Observation,
    StepResult,
    TaskConfig,
)


class JSONEnvironment(BaseEnvironment):
    """Pure Python state machine environment.

    The agent interacts via API_CALL actions with operations like
    add_todo, complete_todo, delete_todo. Observation returns the
    full state as a JSON dict.
    """

    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__(task_config)
        self._state: dict[str, Any] = {"todos": []}
        self._action_history: list[dict[str, Any]] = []

    @property
    def env_id(self) -> str:
        return "json"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def _reset(self) -> Observation:
        self._state = {"todos": []}
        self._action_history = []
        return self._build_observation()

    async def _step(self, action: Action) -> StepResult:
        info: dict[str, Any] = {"action": action.model_dump(), "valid": True, "error": None}
        reward = 0.0

        if action.type != ActionType.API_CALL:
            info["valid"] = False
            info["error"] = f"JSON env only accepts api_call actions, got {action.type}"
            reward = -0.1
        else:
            result = self._execute_operation(action.value or "", action.params or {})
            if result["success"]:
                reward = 0.05
            else:
                info["valid"] = False
                info["error"] = result["error"]
                reward = -0.1

        self._action_history.append(
            {
                "step": self.step_count,
                "action": action.model_dump(),
                "reward": reward,
                "valid": info["valid"],
            }
        )

        # Check goal
        goal_met = self._check_goal()
        if goal_met:
            reward += 1.0

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            terminated=goal_met,
            truncated=False,  # Base class handles truncation
            info=info,
        )

    async def _observe(self) -> Observation:
        return self._build_observation()

    async def evaluate(self) -> dict[str, float]:
        goal_met = self._check_goal()
        optimal = self.task_config.optimal_steps or self.max_steps
        efficiency = (
            max(0.0, 1.0 - (self.step_count - optimal) / self.max_steps) if goal_met else 0.0
        )
        invalid = sum(1 for a in self._action_history if not a.get("valid", True))

        return {
            "completion": 1.0 if goal_met else 0.0,
            "efficiency": round(efficiency, 3),
            "accuracy": 1.0 if goal_met else self._partial_accuracy(),
            "safety": round(max(0.0, 1.0 - (invalid * 0.15)), 3),
            "total_steps": self.step_count,
            "optimal_steps": optimal,
            "invalid_actions": invalid,
        }

    async def cleanup(self) -> None:
        """No resources to release for JSON env. Idempotent."""
        self._state = {"todos": []}

    # ------------------------------------------------------------------
    # Internal operations
    # ------------------------------------------------------------------

    def _execute_operation(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a state mutation operation."""
        if operation == "add_todo":
            title = params.get("title", "")
            if not title:
                return {"success": False, "error": "add_todo requires a 'title' param"}
            todo = {
                "id": uuid.uuid4().hex[:8],
                "title": title,
                "completed": False,
            }
            self._state["todos"].append(todo)
            return {"success": True}

        if operation == "complete_todo":
            todo_id = params.get("id", "")
            for todo in self._state["todos"]:
                if todo["id"] == todo_id:
                    todo["completed"] = True
                    return {"success": True}
            return {"success": False, "error": f"Todo '{todo_id}' not found"}

        if operation == "delete_todo":
            todo_id = params.get("id", "")
            before = len(self._state["todos"])
            self._state["todos"] = [t for t in self._state["todos"] if t["id"] != todo_id]
            if len(self._state["todos"]) < before:
                return {"success": True}
            return {"success": False, "error": f"Todo '{todo_id}' not found"}

        return {"success": False, "error": f"Unknown operation: '{operation}'"}

    def _check_goal(self) -> bool:
        """Check if the task goal is met by inspecting state."""
        goal = self.task_config.goal
        goal_type = goal.get("type")

        if goal_type == "element_count":
            field = str(goal.get("field", "todos"))
            expected_count = int(goal.get("count", 0))
            actual = self._state.get(field, [])
            return bool(len(actual) == expected_count)

        return False

    def _partial_accuracy(self) -> float:
        """Partial credit for incomplete goals."""
        goal = self.task_config.goal
        if goal.get("type") == "element_count":
            field = str(goal.get("field", "todos"))
            expected = int(goal.get("count", 0))
            actual = len(self._state.get(field, []))
            if expected > 0:
                return float(round(min(actual, expected) / expected, 3))
        return 0.0

    def _build_observation(self) -> Observation:
        """Build an Observation from current state."""
        return Observation(
            state=self._state,
            task_description=self.task_config.description,
            available_action_types=["api_call"],
            step_count=self.step_count,
            max_steps=self.max_steps,
        )
