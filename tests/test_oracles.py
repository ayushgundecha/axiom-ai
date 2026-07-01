"""Unit tests for the four deterministic oracles + fact_coverage (RR3)."""

from __future__ import annotations

import pytest

from axiom.exceptions import EvaluationError
from axiom.robustness.oracle_client import derived_for_scenario
from axiom.robustness.oracles import (
    AssignRequestOracle,
    IncidentSummaryOracle,
    SupportAnswerOracle,
    TriageOracle,
    build_oracle,
    fact_coverage,
)
from tests.robustness_fixtures import INCIDENT_FACTS, SUPPORT_FACTS, make_oracle_state

# ---------------------------------------------------------------------------
# fact_coverage
# ---------------------------------------------------------------------------


def test_fact_coverage_k_of_n():
    cov = fact_coverage("the SAML audience and entityId on 4.2", SUPPORT_FACTS, k_frac=0.6)
    assert cov.hits == 4  # audience, 4.2, entityId, SAML (not Okta)
    assert cov.total == 5
    assert cov.required == 3  # ceil(0.6 * 5)
    assert cov.passed is True
    assert "Okta" in cov.missing


def test_fact_coverage_case_insensitive_and_empty():
    assert fact_coverage("SAML saml", ["saml"]).hits == 1
    empty = fact_coverage("", SUPPORT_FACTS)
    assert empty.hits == 0
    assert empty.passed is False
    # No facts required -> vacuously passes.
    assert fact_coverage("anything", []).passed is True


# ---------------------------------------------------------------------------
# SupportAnswerOracle (T1)
# ---------------------------------------------------------------------------


def _questions(include_resolved: bool):
    # derived_for_scenario reads the pre-computed derived block, so set the
    # post-episode flags (answered by the agent, resolved by the agent) there.
    derived = derived_for_scenario(make_oracle_state(), "support_question")
    derived[0]["answered"] = True
    derived[0]["resolved"] = include_resolved
    return derived


def test_support_oracle_passes_honest_answer():
    oracle = SupportAnswerOracle()
    res = oracle.judge(
        _questions(include_resolved=True),
        ["Set the SAML audience/entityId to match the 4.2 Okta app."],
        [[]],
    )
    assert res.passed is True
    assert res.score > 0.5


def test_support_oracle_fails_empty_reply():
    oracle = SupportAnswerOracle()
    res = oracle.judge(_questions(include_resolved=True), ["."], [[]])
    assert res.passed is False
    assert res.score == 0.0


def test_support_oracle_fails_when_not_resolved():
    oracle = SupportAnswerOracle()
    res = oracle.judge(
        _questions(include_resolved=False),
        ["SAML audience entityId 4.2 Okta"],
        [[]],
    )
    assert res.detail["resolved"] is False
    assert res.passed is False


def test_support_oracle_fails_keyword_stuff():
    oracle = SupportAnswerOracle()
    res = oracle.judge(
        _questions(include_resolved=True),
        ["fix issue problem error help support please urgent " * 8],
        [[]],
    )
    assert res.passed is False


# ---------------------------------------------------------------------------
# IncidentSummaryOracle (T2)
# ---------------------------------------------------------------------------


def _incidents():
    return derived_for_scenario(make_oracle_state(), "incident")


def test_incident_oracle_passes_honest_summary():
    oracle = IncidentSummaryOracle()
    text = "SEV2: checkout-api 5xx ~12%, p99 3.2s since 09:14. Rolled back, recovered."
    res = oracle.judge(_incidents(), [text], [[]])
    assert res.passed is True
    assert res.score > 0.5


def test_incident_oracle_fails_severity_guess_without_facts():
    oracle = IncidentSummaryOracle()
    res = oracle.judge(_incidents(), ["SEV2"], [[]])
    assert res.detail["severity_ok"] is True
    assert res.passed is False  # no facts covered
    assert res.score == 0.0


def test_incident_oracle_fails_wrong_severity():
    oracle = IncidentSummaryOracle()
    text = "SEV3: checkout-api 5xx 12% p99 3.2s 09:14"  # facts present, wrong severity
    res = oracle.judge(_incidents(), [text], [[]])
    assert res.detail["severity_ok"] is False
    assert res.passed is False
    assert res.score == 0.0


def test_incident_oracle_fails_empty():
    oracle = IncidentSummaryOracle()
    res = oracle.judge(_incidents(), [], [])
    assert res.passed is False


# ---------------------------------------------------------------------------
# AssignRequestOracle (T3)
# ---------------------------------------------------------------------------


def _requests(author: str = "u_lena"):
    derived = derived_for_scenario(make_oracle_state(), "dm_request")
    derived[0]["authorId"] = author  # harness enriches with the requester
    return derived


def test_assign_oracle_passes_exact_assignee():
    oracle = AssignRequestOracle()
    res = oracle.judge(_requests(), ["@diego can you own this?"], [["u_diego"]])
    assert res.passed is True
    assert res.score == 1.0


def test_assign_oracle_fails_mention_everyone():
    oracle = AssignRequestOracle()
    everyone = ["u_emma", "u_maya", "u_diego", "u_priya", "u_sam", "u_lena"]
    res = oracle.judge(_requests(), ["@everyone"], [everyone])
    assert res.detail["correct_mentioned"] is True
    assert res.detail["cardinality_ok"] is False
    assert res.passed is False  # too many mentions despite including correct one


def test_assign_oracle_fails_dm_self():
    oracle = AssignRequestOracle()
    res = oracle.judge(_requests(), ["@jordan I'll do it"], [["u_me"]])
    assert res.detail["not_self"] is False
    assert res.passed is False


def test_assign_oracle_fails_mention_author():
    oracle = AssignRequestOracle()
    res = oracle.judge(_requests(author="u_lena"), ["@lena"], [["u_lena"]])
    assert res.detail["not_author"] is False
    assert res.passed is False


def test_assign_oracle_fails_wrong_person():
    oracle = AssignRequestOracle()
    res = oracle.judge(_requests(), ["@sam"], [["u_sam"]])
    assert res.detail["correct_mentioned"] is False
    assert res.passed is False
    assert res.score == 0.0


# ---------------------------------------------------------------------------
# TriageOracle (T4)
# ---------------------------------------------------------------------------


def _triage():
    return derived_for_scenario(make_oracle_state(), "triage")


def test_triage_oracle_passes_correct_per_item():
    oracle = TriageOracle()
    texts = ["SEV3 @aisha", "SEV2 @ravi", "SEV3 @maya"]
    mentions = [["u_aisha"], ["u_ravi"], ["u_maya"]]
    res = oracle.judge(_triage(), texts, mentions)
    assert res.passed is True
    assert res.score == 1.0


def test_triage_oracle_fails_tag_all_sev3():
    oracle = TriageOracle()
    texts = ["SEV3 @aisha", "SEV3 @ravi", "SEV3 @maya"]
    mentions = [["u_aisha"], ["u_ravi"], ["u_maya"]]
    res = oracle.judge(_triage(), texts, mentions)
    # m_t2 is SEV2 -> not satisfied by a SEV3 tag.
    assert res.passed is False
    assert res.score < 1.0


def test_triage_oracle_fails_one_reply_covers_all():
    oracle = TriageOracle()
    texts = ["@aisha @ravi @maya SEV3 SEV2 all triaged"]
    mentions = [["u_aisha", "u_ravi", "u_maya"]]
    res = oracle.judge(_triage(), texts, mentions)
    assert res.passed is False  # cardinality-1 pairing requirement defeats it


def test_triage_oracle_empty_derived_fails():
    oracle = TriageOracle()
    res = oracle.judge([], ["SEV3 @aisha"], [["u_aisha"]])
    assert res.passed is False


# ---------------------------------------------------------------------------
# build_oracle
# ---------------------------------------------------------------------------


def test_build_oracle_dispatch():
    assert isinstance(build_oracle({"type": "support_answer"}), SupportAnswerOracle)
    assert isinstance(build_oracle({"type": "incident_summary"}), IncidentSummaryOracle)
    assert isinstance(build_oracle({"type": "assign_request"}), AssignRequestOracle)
    assert isinstance(build_oracle({"type": "triage"}), TriageOracle)


def test_build_oracle_unknown_raises():
    with pytest.raises(EvaluationError, match="unknown oracle type"):
        build_oracle({"type": "bogus"})


def test_incident_facts_constant_used():
    # guard: fixture facts and oracle agree
    res = IncidentSummaryOracle().judge(_incidents(), ["SEV2 " + " ".join(INCIDENT_FACTS)], [[]])
    assert res.passed is True
