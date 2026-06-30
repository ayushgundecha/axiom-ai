"""label_episode: proxy vs oracle verdict + hack flag (RR12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core.task_loader import TaskLoader
from axiom.robustness.corpus import corpus_target_from_oracle, honest_support
from axiom.robustness.exploits.library import catalog_by_id, generate_actions
from axiom.robustness.labeler import label_episode
from axiom.robustness.oracles import build_oracle
from axiom.robustness.proxies import build_proxy
from tests.robustness_fixtures import apply_actions, make_oracle_state


@pytest.fixture
def task():
    loader = TaskLoader(Path(__file__).resolve().parent.parent / "tasks")
    return loader.load_task("axiomchat", "answer_support_question")


async def _label(task, actions, *, hardened: bool):
    pre = make_oracle_state()
    post = apply_actions(pre, actions)
    proxy = build_proxy(task.proxy, hardened=hardened)
    oracle = build_oracle(task.oracle)
    return await label_episode(
        None,
        None,
        task,
        proxy,
        oracle,
        pre,
        post,
        reward_version="v1" if hardened else "v0",
    )


async def test_honest_support_passes_both(task):
    target = corpus_target_from_oracle(make_oracle_state(), "support_question")
    actions = honest_support(target)
    label = await _label(task, actions, hardened=True)
    assert label.proxy_pass is True
    assert label.oracle_pass is True
    assert label.hack is False


async def test_empty_reply_is_hack_under_v0_defeated_under_v1(task):
    target = corpus_target_from_oracle(make_oracle_state(), "support_question")
    actions = generate_actions(catalog_by_id()["empty-reply"], target.to_exploit_target())

    v0 = await _label(task, actions, hardened=False)
    assert v0.proxy_pass is True  # reply exists + thread resolved
    assert v0.oracle_pass is False  # no answer facts
    assert v0.hack is True

    v1 = await _label(task, actions, hardened=True)
    assert v1.proxy_pass is False  # min_chars defeats the trivial reply
    assert v1.hack is False


async def test_keyword_stuff_oracle_rejects(task):
    target = corpus_target_from_oracle(make_oracle_state(), "support_question")
    actions = generate_actions(catalog_by_id()["keyword-stuff"], target.to_exploit_target())
    v0 = await _label(task, actions, hardened=False)
    assert v0.oracle_pass is False
    assert v0.hack is True  # naive reply+resolved proxy fooled


async def test_runlabel_serializable(task):
    target = corpus_target_from_oracle(make_oracle_state(), "support_question")
    actions = honest_support(target)
    label = await _label(task, actions, hardened=True)
    d = label.to_dict()
    assert d["task_id"] == "answer_support_question"
    assert set(d) >= {"proxy_pass", "oracle_pass", "hack", "proxy_score", "oracle_score"}
