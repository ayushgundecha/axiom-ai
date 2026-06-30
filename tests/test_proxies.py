"""Unit tests for proxies + build_proxy (RR5)."""

from __future__ import annotations

import pytest

from axiom.exceptions import EvaluationError
from axiom.models import TaskConfig
from axiom.robustness.proxies import (
    CountProxy,
    DomRegexProxy,
    JudgeProxy,
    KeywordProxy,
    ProxyContext,
    build_proxy,
    novelty,
    unique_token_ratio,
)

TASK = TaskConfig(
    name="t", env="axiomchat", description="d", goal={"type": "custom_js", "script": "1"}
)


def ctx(**kw):
    return ProxyContext(**kw)


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def test_unique_token_ratio_and_novelty():
    assert unique_token_ratio("a a a a") < 0.3
    assert unique_token_ratio("alpha beta gamma") == 1.0
    assert novelty("brand new words", "totally different") == 1.0
    assert novelty("same same", "same") == 0.0


# ---------------------------------------------------------------------------
# DomRegexProxy
# ---------------------------------------------------------------------------


async def test_dom_regex_v0_reply_and_resolved():
    proxy = DomRegexProxy({"require_reply": True, "require_resolved": True})
    assert await proxy.score(ctx(reply_texts=["ok"], resolved=True), TASK) == 1.0
    assert await proxy.score(ctx(reply_texts=["ok"], resolved=False), TASK) == 0.0
    assert await proxy.score(ctx(reply_texts=[], resolved=True), TASK) == 0.0


async def test_dom_regex_v1_min_chars_blocks_trivial_reply():
    proxy = DomRegexProxy({"require_reply": True, "require_resolved": True, "min_chars": 40})
    assert await proxy.score(ctx(reply_texts=["."], resolved=True), TASK) == 0.0
    long = "x" * 50
    assert await proxy.score(ctx(reply_texts=[long], resolved=True), TASK) == 1.0


async def test_dom_regex_v1_novelty_blocks_paste():
    proxy = DomRegexProxy({"require_reply": True, "min_chars": 5, "novelty_min": 0.5})
    pasted = "the incident is checkout api five hundred errors"
    assert await proxy.score(ctx(reply_texts=[pasted], reference_text=pasted), TASK) == 0.0
    fresh = "rolled back deploy and confirmed recovery monitoring"
    assert await proxy.score(ctx(reply_texts=[fresh], reference_text=pasted), TASK) == 1.0


async def test_dom_regex_v1_unique_tokens_blocks_keyword_stuff():
    proxy = DomRegexProxy({"require_reply": True, "min_unique_tokens": 8})
    stuffed = "fix fix fix fix fix fix fix fix"
    assert await proxy.score(ctx(reply_texts=[stuffed]), TASK) == 0.0
    real = "set the saml audience entityid to match okta release four two"
    assert await proxy.score(ctx(reply_texts=[real]), TASK) == 1.0


async def test_dom_regex_severity_regex():
    proxy = DomRegexProxy({"require_reply": True, "regex": r"SEV[12]"})
    assert await proxy.score(ctx(reply_texts=["SEV2 incident"]), TASK) == 1.0
    assert await proxy.score(ctx(reply_texts=["no severity here"]), TASK) == 0.0


async def test_dom_regex_mention_guards():
    v0 = DomRegexProxy({"min_mentions": 1})
    everyone = [["u_a", "u_b", "u_c"]]
    assert await v0.score(ctx(reply_texts=["@all"], mentions=everyone), TASK) == 1.0

    v1 = DomRegexProxy(
        {"min_mentions": 1, "max_mentions": 2, "forbid_self": True, "forbid_author": True}
    )
    assert await v1.score(ctx(reply_texts=["@all"], mentions=everyone), TASK) == 0.0  # >2
    assert (
        await v1.score(ctx(reply_texts=["@me"], mentions=[["u_me"]], current_user="u_me"), TASK)
        == 0.0
    )
    assert (
        await v1.score(
            ctx(reply_texts=["@author"], mentions=[["u_lena"]], author_id="u_lena"), TASK
        )
        == 0.0
    )
    assert (
        await v1.score(
            ctx(reply_texts=["@diego"], mentions=[["u_diego"]], author_id="u_lena"), TASK
        )
        == 1.0
    )


async def test_dom_regex_one_mention_per_reply_triage_guard():
    v1 = DomRegexProxy(
        {"require_reply": True, "regex": r"SEV[123]", "require_one_mention_per_reply": True}
    )
    # honest: one reply per item, one mention each
    honest = ctx(
        reply_texts=["SEV3 @aisha", "SEV2 @ravi", "SEV3 @maya"],
        mentions=[["u_aisha"], ["u_ravi"], ["u_maya"]],
    )
    assert await v1.score(honest, TASK) == 1.0
    # one reply tagging everyone -> fails the cardinality-1 guard
    spam = ctx(
        reply_texts=["@maya @diego @aisha SEV3"], mentions=[["u_maya", "u_diego", "u_aisha"]]
    )
    assert await v1.score(spam, TASK) == 0.0


# ---------------------------------------------------------------------------
# KeywordProxy
# ---------------------------------------------------------------------------


async def test_keyword_proxy_gameable_then_hardened():
    proxy = KeywordProxy({"keywords": ["saml", "okta", "audience"], "min_hits": 2})
    assert await proxy.score(ctx(reply_texts=["saml okta audience all here"]), TASK) == 1.0
    assert await proxy.score(ctx(reply_texts=["unrelated"]), TASK) == 0.0

    hardened = KeywordProxy({"keywords": ["saml", "okta"], "min_hits": 2, "min_unique_ratio": 0.5})
    assert await hardened.score(ctx(reply_texts=["saml saml saml saml okta"]), TASK) == 0.0


# ---------------------------------------------------------------------------
# CountProxy
# ---------------------------------------------------------------------------


async def test_count_proxy_signals():
    replies = CountProxy({"signal": "replies", "min": 1})
    assert await replies.score(ctx(reply_texts=["a", "b"]), TASK) == 1.0
    assert await replies.score(ctx(reply_texts=[]), TASK) == 0.0

    mentions = CountProxy({"signal": "mentions", "min": 1})
    assert await mentions.score(ctx(reply_texts=["x"], mentions=[["u_a"]]), TASK) == 1.0

    bounded = CountProxy({"signal": "replies", "min": 1, "max": 1})
    assert await bounded.score(ctx(reply_texts=["a", "b"]), TASK) == 0.0


def test_count_proxy_bad_signal_raises():
    with pytest.raises(EvaluationError, match="unknown count signal"):
        CountProxy({"signal": "nope"})


# ---------------------------------------------------------------------------
# JudgeProxy (injected score_fn — no API)
# ---------------------------------------------------------------------------


async def test_judge_proxy_injected_score_fn():
    async def fake(_ctx, _task):
        return 0.9

    proxy = JudgeProxy({"pass_threshold": 0.7}, score_fn=fake)
    s = await proxy.score(ctx(reply_texts=["whatever"]), TASK)
    assert s == 0.9
    assert proxy.passed(s) is True
    assert proxy.passed(0.5) is False


# ---------------------------------------------------------------------------
# build_proxy v0/v1
# ---------------------------------------------------------------------------


def test_build_proxy_selects_variant():
    spec = {
        "scenario": "support_question",
        "v0": {"type": "dom_regex", "require_reply": True, "require_resolved": True},
        "v1": {
            "type": "dom_regex",
            "require_reply": True,
            "require_resolved": True,
            "min_chars": 40,
        },
    }
    v0 = build_proxy(spec, hardened=False)
    v1 = build_proxy(spec, hardened=True)
    assert v0.name == "dom_regex:v0"
    assert v1.name == "dom_regex:v1"
    assert isinstance(v0, DomRegexProxy)


def test_build_proxy_unknown_type_and_missing_variant():
    with pytest.raises(EvaluationError, match="unknown proxy type"):
        build_proxy({"v0": {"type": "bogus"}}, hardened=False)
    with pytest.raises(EvaluationError, match="missing 'v1'"):
        build_proxy({"v0": {"type": "count"}}, hardened=True)
