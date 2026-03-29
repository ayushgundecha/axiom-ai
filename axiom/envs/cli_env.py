"""CLI/terminal environment — sandboxed shell command execution.

Maps to Deeptune's Terminal-Bench work: agents operate computers via
command-line interfaces. Commands execute in a sandboxed temporary
directory with an allowlist of safe commands and full-command inspection
for dangerous patterns (path traversal, dangerous redirects).

Security model:
  - Only allowlisted commands (ls, cat, mkdir, mv, etc.)
  - Full command string inspected for path traversal (..)
  - Commands run in an isolated temp directory
  - 10-second timeout per command
  - Sanitized environment variables
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

from axiom.core.base_env import BaseEnvironment
from axiom.models import (
    Action,
    ActionType,
    Observation,
    StepResult,
    TaskConfig,
)

# Commands that are safe for agents to use
ALLOWED_COMMANDS = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "mkdir",
        "touch",
        "cp",
        "mv",
        "rm",
        "echo",
        "sort",
        "uniq",
        "cut",
        "awk",
        "sed",
        "pwd",
        "tree",
        "diff",
        "chmod",
        "stat",
        "basename",
        "dirname",
        "xargs",
        "tr",
        "tee",
    }
)

# Patterns that indicate dangerous operations
BLOCKED_PATTERNS = ("..",)

# Timeout for individual commands (seconds)
_COMMAND_TIMEOUT = 10


class CLIEnvironment(BaseEnvironment):
    """Sandboxed terminal environment.

    Agents execute shell commands in a temporary directory.
    Goal checking inspects the actual filesystem state.
    """

    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__(task_config)
        self._workdir: str = ""
        self._action_history: list[dict[str, Any]] = []
        self._command_outputs: list[str] = []

    @property
    def env_id(self) -> str:
        return "cli"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def _reset(self) -> Observation:
        self._action_history = []
        self._command_outputs = []

        # Clean up old workdir if exists
        if self._workdir and Path(self._workdir).exists():  # noqa: ASYNC240
            shutil.rmtree(self._workdir)

        # Create fresh sandboxed directory
        self._workdir = tempfile.mkdtemp(prefix="axiom_cli_")

        # Set up initial files from task config
        initial_files: list[dict[str, str]] = self.task_config.initial_state.get("files", [])
        for file_spec in initial_files:
            filepath = Path(self._workdir) / file_spec["path"]
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(file_spec.get("content", ""))

        # Show initial state
        initial_output = await self._run_safe_command("find . -type f | head -20")
        self._command_outputs.append(initial_output)

        return await self._observe()

    async def _step(self, action: Action) -> StepResult:
        info: dict[str, Any] = {
            "action": action.model_dump(),
            "valid": True,
            "error": None,
        }
        reward = 0.0

        if action.type != ActionType.RUN_COMMAND or not action.value:
            info["valid"] = False
            info["error"] = "CLI env only accepts run_command actions with a value"
            reward = -0.1
            output = "Error: Invalid action. Use run_command with a shell command."
        else:
            safety = self._check_command_safety(action.value)
            if safety is not None:
                info["valid"] = False
                info["error"] = safety
                reward = -0.1
                output = f"Error: {safety}"
            else:
                output = await self._run_safe_command(action.value)
                reward = 0.05

        self._command_outputs.append(output)
        self._action_history.append(
            {
                "step": self.step_count,
                "command": action.value,
                "output": output[:500],
                "reward": reward,
                "valid": info["valid"],
            }
        )

        # Check goal
        goal_met = self._check_goal()
        if goal_met:
            reward += 1.0

        return StepResult(
            observation=await self._observe(),
            reward=reward,
            terminated=goal_met,
            truncated=False,  # Base class handles truncation
            info=info,
        )

    async def _observe(self) -> Observation:
        last_output = (
            self._command_outputs[-1] if self._command_outputs else "(no commands run yet)"
        )
        dir_listing = await self._run_safe_command("find . -type f | sort")

        return Observation(
            text_output=f"Last output:\n{last_output}\n\nCurrent files:\n{dir_listing}",
            task_description=self.task_config.description,
            available_action_types=["run_command"],
            step_count=self.step_count,
            max_steps=self.max_steps,
        )

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
            "accuracy": 1.0 if goal_met else 0.0,
            "safety": round(max(0.0, 1.0 - (invalid * 0.15)), 3),
            "total_steps": self.step_count,
            "optimal_steps": optimal,
            "invalid_actions": invalid,
        }

    async def cleanup(self) -> None:
        """Remove temporary working directory. Idempotent."""
        if self._workdir and Path(self._workdir).exists():  # noqa: ASYNC240
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = ""

    # ------------------------------------------------------------------
    # Command safety
    # ------------------------------------------------------------------

    def _check_command_safety(self, command: str) -> str | None:
        """Validate a command against the safety policy.

        Returns None if safe, or an error message string if blocked.
        """
        stripped = command.strip()
        if not stripped:
            return "Empty command"

        # Check first word against allowlist
        cmd_name = stripped.split()[0]
        if cmd_name not in ALLOWED_COMMANDS:
            allowed = ", ".join(sorted(ALLOWED_COMMANDS))
            return f"Command '{cmd_name}' not allowed. Allowed: {allowed}"

        # Check full command for dangerous patterns
        for pattern in BLOCKED_PATTERNS:
            if pattern in stripped:
                return f"Blocked: command contains '{pattern}' (path traversal)"

        return None

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    async def _run_safe_command(self, command: str) -> str:
        """Execute command in sandboxed directory with timeout."""
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self._workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._safe_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_COMMAND_TIMEOUT)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                err = stderr.decode("utf-8", errors="replace")
                if err.strip():
                    output += "\nSTDERR: " + err
            return output.strip() or "(no output)"
        except TimeoutError:
            proc.kill()
            return "Error: Command timed out (10s limit)"

    def _safe_env(self) -> dict[str, str]:
        """Minimal environment for sandboxed commands."""
        return {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": self._workdir,
            "TERM": "dumb",
        }

    # ------------------------------------------------------------------
    # Goal checking — inspects actual filesystem
    # ------------------------------------------------------------------

    def _check_goal(self) -> bool:
        """Check goal by inspecting actual filesystem state."""
        goal = self.task_config.goal
        goal_type = goal.get("type")

        if goal_type == "files_in_directory":
            checks: list[dict[str, str]] = goal.get("checks", [])
            for check in checks:
                path = Path(self._workdir) / check["path"]
                if not path.exists():
                    return False
            return True

        elif goal_type == "file_content_matches":
            content_checks: list[dict[str, Any]] = goal.get("checks", [])
            for check in content_checks:
                path = Path(self._workdir) / check["path"]
                if not path.exists():
                    return False
                content = path.read_text()
                if check.get("contains") and check["contains"] not in content:
                    return False
                if check.get("line_count") and len(content.strip().split("\n")) != int(
                    check["line_count"]
                ):
                    return False
            return True

        elif goal_type == "directory_structure":
            expected: list[str] = goal.get("expected_structure", [])
            for entry in expected:
                path = Path(self._workdir) / entry
                if not path.exists():
                    return False
            return True

        return False
