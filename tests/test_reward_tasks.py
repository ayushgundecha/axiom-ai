"""The 4 reward task YAMLs load + their proxy/oracle blocks build (RR6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core.task_loader import TaskLoader
from axiom.robustness.oracles import (
    AssignRequestOracle,
    IncidentSummaryOracle,
    SupportAnswerOracle,
    TriageOracle,
    build_oracle,
)
from axiom.robustness.proxies import build_proxy

REWARD_TASKS = [
    "answer_support_question",
    "summarize_incident",
    "assign_request",
    "triage_backlog",
]

EXPECTED_ORACLE = {
    "answer_support_question": SupportAnswerOracle,
    "summarize_incident": IncidentSummaryOracle,
    "assign_request": AssignRequestOracle,
    "triage_backlog": TriageOracle,
}


@pytest.fixture
def loader() -> TaskLoader:
    return TaskLoader(Path(__file__).resolve().parent.parent / "tasks")


@pytest.mark.parametrize("task_id", REWARD_TASKS)
def test_reward_task_loads_with_proxy_and_oracle(loader: TaskLoader, task_id: str) -> None:
    task = loader.load_task("axiomchat", task_id)
    assert task.seed is not None
    assert task.scale == "medium"
    assert task.proxy is not None
    assert task.oracle is not None
    # proxy has both v0 (naive) and v1 (hardened) variants
    assert "v0" in task.proxy and "v1" in task.proxy
    assert "scenario" in task.proxy


@pytest.mark.parametrize("task_id", REWARD_TASKS)
def test_reward_task_builds_proxy_versions(loader: TaskLoader, task_id: str) -> None:
    task = loader.load_task("axiomchat", task_id)
    assert task.proxy is not None
    v0 = build_proxy(task.proxy, hardened=False)
    v1 = build_proxy(task.proxy, hardened=True)
    assert v0.name.endswith(":v0")
    assert v1.name.endswith(":v1")


@pytest.mark.parametrize("task_id", REWARD_TASKS)
def test_reward_task_builds_oracle(loader: TaskLoader, task_id: str) -> None:
    task = loader.load_task("axiomchat", task_id)
    assert task.oracle is not None
    oracle = build_oracle(task.oracle)
    assert isinstance(oracle, EXPECTED_ORACLE[task_id])


def test_v1_is_at_least_as_strict_as_v0_support(loader: TaskLoader) -> None:
    # The hardened support proxy must add gates the naive one lacks.
    task = loader.load_task("axiomchat", "answer_support_question")
    assert task.proxy is not None
    assert task.proxy["v1"].get("min_chars", 0) > task.proxy["v0"].get("min_chars", 0)
