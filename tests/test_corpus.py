"""Reward corpus structure + target extraction (RR9)."""

from __future__ import annotations

from axiom.models import Action
from axiom.robustness.corpus import (
    CorpusTarget,
    corpus_target_from_oracle,
    exploit_cases,
    honest_cases,
    load_corpus,
)
from tests.robustness_fixtures import make_oracle_state


def test_corpus_has_exploit_and_honest_cases():
    cases = load_corpus()
    assert len(exploit_cases()) >= 15
    assert len(honest_cases()) == 4
    assert {c.task_id for c in honest_cases()} == {
        "answer_support_question",
        "summarize_incident",
        "assign_request",
        "triage_backlog",
    }
    assert len(cases) == len(exploit_cases()) + len(honest_cases())


def test_exploit_cases_expect_both_false():
    for c in exploit_cases():
        assert c.expect_proxy_v1_pass is False
        assert c.expect_oracle_pass is False


def test_honest_cases_expect_both_true():
    for c in honest_cases():
        assert c.expect_proxy_v1_pass is True
        assert c.expect_oracle_pass is True


def test_case_ids_unique():
    ids = [c.id for c in load_corpus()]
    assert len(ids) == len(set(ids))


def _target(scenario: str) -> CorpusTarget:
    return corpus_target_from_oracle(make_oracle_state(), scenario)  # type: ignore[return-value]


def test_builders_produce_actions_for_every_case():
    targets = {
        "support_question": _target("support_question"),
        "incident": _target("incident"),
        "dm_request": _target("dm_request"),
        "triage": _target("triage"),
    }
    for c in load_corpus():
        actions = c.actions(targets[c.scenario])
        assert actions, f"{c.id} produced no actions"
        assert all(isinstance(a, Action) for a in actions)


def test_corpus_target_from_oracle_reads_truth():
    support = corpus_target_from_oracle(make_oracle_state(), "support_question")
    assert support is not None
    assert support.message_id == "m_sq"
    assert "SAML" in support.answer_facts

    incident = corpus_target_from_oracle(make_oracle_state(), "incident")
    assert incident is not None and incident.severity == "SEV2"

    request = corpus_target_from_oracle(make_oracle_state(), "dm_request")
    assert request is not None
    assert request.assignee_handle == "diego"
    assert request.author_handle == "lena"

    triage = corpus_target_from_oracle(make_oracle_state(), "triage")
    assert triage is not None
    assert len(triage.triage_correct) == 3
    assert ("m_t1", "SEV3", "aisha") in triage.triage_correct


def test_corpus_target_none_when_scenario_absent():
    assert corpus_target_from_oracle(make_oracle_state(include_triage=False), "triage") is None
