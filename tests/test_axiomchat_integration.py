"""Integration test: a scripted agent completes an AxiomChat task end-to-end.

Requires the AxiomChat app running (docker-compose up axiomchat-app, or run the
server locally on :3100). Marked @pytest.mark.integration so it is excluded from
the default unit run. Uses deterministic scripted actions rather than the live
Claude API so it is reproducible in CI; the same actions a Claude agent would
take complete the post_message task.
"""

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import requires_axiomchat

CHANNEL_GENERAL = "[data-testid='channel-link-c_general']"
MESSAGE_INPUT = "[data-testid='message-input']"
SEND_BUTTON = "[data-testid='send-button']"
POST_TEXT = "Deploying v4.3 to production now"


@pytest.mark.integration
class TestAxiomChatIntegration:
    @requires_axiomchat
    @pytest.mark.asyncio
    async def test_scripted_agent_posts_message_and_records_trajectory(
        self, sample_axiomchat_task_config: dict[str, Any], tmp_path: Path
    ) -> None:
        from axiom.core.trajectory import TrajectoryRecorder
        from axiom.envs.axiomchat_env import AxiomChatEnvironment
        from axiom.models import Action, ActionType, TaskConfig

        config = TaskConfig(**sample_axiomchat_task_config)
        recorder = TrajectoryRecorder()
        session_id = "itest_axiomchat_post"
        recorder.start_session(session_id, config.name, "axiomchat")

        actions = [
            Action(type=ActionType.CLICK, selector=CHANNEL_GENERAL),
            Action(type=ActionType.TYPE, selector=MESSAGE_INPUT, value=POST_TEXT),
            Action(type=ActionType.CLICK, selector=SEND_BUTTON),
        ]

        async with AxiomChatEnvironment(config) as env:
            await env.reset()

            result = None
            for step_num, action in enumerate(actions, start=1):
                result = await env.step(action)
                recorder.record_step(
                    session_id,
                    step_num,
                    action.model_dump(),
                    result.observation.model_dump(),
                    result.reward,
                    result.terminated,
                    result.truncated,
                )

            assert result is not None
            assert result.terminated, "post_message goal should be met after sending"

            scores = await env.evaluate()
            recorder.set_evaluation(session_id, scores)
            assert scores["completion"] == 1.0

        # Trajectory persisted to disk (JSON with no inline base64).
        out_dir = recorder.save(session_id, tmp_path)
        assert out_dir.exists()
        trajectory_json = out_dir / "trajectory.json"
        assert trajectory_json.exists()
        import json

        saved = json.loads(trajectory_json.read_text())
        assert saved["env_type"] == "axiomchat"
        assert saved["total_steps"] == len(actions)
        assert saved["evaluation"]["completion"] == 1.0
        # Screenshots are stored as separate PNGs, never inline base64 in the JSON.
        assert "screenshot_base64" not in trajectory_json.read_text()
        assert (out_dir / "screenshots").exists()

    @requires_axiomchat
    @pytest.mark.asyncio
    async def test_reset_is_deterministic_for_seed(
        self, sample_axiomchat_task_config: dict[str, Any]
    ) -> None:
        """Two resets at the same seed observe the same workspace."""
        from axiom.envs.axiomchat_env import AxiomChatEnvironment
        from axiom.models import TaskConfig

        config = TaskConfig(**sample_axiomchat_task_config)
        async with AxiomChatEnvironment(config) as env:
            obs1 = await env.reset()
            obs2 = await env.reset()
            # The simplified DOM (channel list + messages) is identical per seed.
            assert obs1.dom_tree == obs2.dom_tree
