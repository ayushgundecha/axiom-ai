"""Unit tests for agent_artifacts — the baseline-diff (RR2, riskiest piece)."""

from __future__ import annotations

from axiom.robustness.oracle_client import (
    agent_artifacts,
    artifact_mentions,
    artifact_texts,
    mention_set,
    message_ids,
)
from tests.robustness_fixtures import (
    agent_reply,
    make_oracle_state,
    public_state,
    with_agent_messages,
)


def test_diff_returns_only_new_me_authored_messages():
    pre = make_oracle_state()
    reply = agent_reply(
        "m_new1",
        "c_support",
        "Set the SAML audience/entityId to match the 4.2 Okta app — that fixes it.",
        thread_root_id="m_sq",
    )
    post = with_agent_messages(pre, [reply], resolve_target="m_sq")

    arts = agent_artifacts(pre, post, current_user="u_me")
    assert len(arts) == 1
    assert arts[0].message_id == "m_new1"
    assert arts[0].thread_root_id == "m_sq"
    assert arts[0].channel_id == "c_support"
    assert "SAML" in arts[0].text


def test_excludes_pre_existing_and_other_authors():
    pre = make_oracle_state()
    me_msg = agent_reply("m_me", "c_support", "my reply", thread_root_id="m_sq")
    other_msg = agent_reply(
        "m_other", "c_support", "not me", thread_root_id="m_sq", author_id="u_maya"
    )
    post = with_agent_messages(pre, [me_msg, other_msg])

    arts = agent_artifacts(pre, post)
    ids = [a.message_id for a in arts]
    assert ids == ["m_me"]  # pre-existing messages and u_maya's reply excluded


def test_root_and_channel_filters():
    pre = make_oracle_state()
    in_thread = agent_reply("m_a", "c_support", "thread reply", thread_root_id="m_sq")
    top_level = agent_reply("m_b", "c_support", "top level post")
    in_incidents = agent_reply("m_c", "c_incidents", "incident reply", thread_root_id="m_inc")
    post = with_agent_messages(pre, [in_thread, top_level, in_incidents])

    assert [a.message_id for a in agent_artifacts(pre, post, root_id="m_sq")] == ["m_a"]
    assert [a.message_id for a in agent_artifacts(pre, post, root_id="m_inc")] == ["m_c"]
    assert {a.message_id for a in agent_artifacts(pre, post, channel_id="c_support")} == {
        "m_a",
        "m_b",
    }
    assert [a.message_id for a in agent_artifacts(pre, post, channel_id="c_incidents")] == ["m_c"]


def test_mentions_carried_through():
    pre = make_oracle_state()
    reply = agent_reply("m_assign", "dm_u_lena", "@diego can you own this?", thread_root_id="m_req")
    post = with_agent_messages(pre, [reply])

    arts = agent_artifacts(pre, post, root_id="m_req")
    assert arts[0].mentions == ("u_diego",)
    assert mention_set(arts) == {"u_diego"}
    assert artifact_mentions(arts) == [["u_diego"]]


def test_mention_everyone_carries_all_resolved_handles():
    pre = make_oracle_state()
    text = "@emma @maya @diego @priya @sam @lena @tom @aisha @ravi @carlos please look"
    reply = agent_reply("m_all", "dm_u_lena", text, thread_root_id="m_req")
    post = with_agent_messages(pre, [reply])

    arts = agent_artifacts(pre, post, root_id="m_req")
    # @everyone/@channel do NOT resolve; only real handles in the roster do.
    assert len(mention_set(arts)) >= 8
    assert "u_diego" in mention_set(arts)


def test_works_on_public_state_too():
    pre = make_oracle_state()
    reply = agent_reply("m_pub", "c_support", "public diff works", thread_root_id="m_sq")
    post = with_agent_messages(pre, [reply])

    arts = agent_artifacts(public_state(pre), public_state(post))
    assert [a.message_id for a in arts] == ["m_pub"]


def test_deterministic_and_order_preserving():
    pre = make_oracle_state()
    replies = [
        agent_reply("m_1", "c_support", "first", thread_root_id="m_sq"),
        agent_reply("m_2", "c_support", "second", thread_root_id="m_sq"),
        agent_reply("m_3", "c_support", "third", thread_root_id="m_sq"),
    ]
    post = with_agent_messages(pre, replies)

    first = agent_artifacts(pre, post)
    second = agent_artifacts(pre, post)
    assert first == second
    assert artifact_texts(first) == ["first", "second", "third"]


def test_no_new_messages_returns_empty():
    pre = make_oracle_state()
    post = with_agent_messages(pre, [])  # agent did nothing
    assert agent_artifacts(pre, post) == []


def test_message_ids_helper():
    state = make_oracle_state()
    ids = message_ids(state)
    assert "m_sq" in ids and "m_inc" in ids and "m_req" in ids
