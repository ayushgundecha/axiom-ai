"""Robustness report schema + summary/leaderboard (RR15)."""

from __future__ import annotations

import json

from axiom.robustness.labeler import RunLabel
from axiom.robustness.report import build_report, build_summary, write_report


def _run(task, version, agent, *, proxy_pass, oracle_pass):
    return RunLabel(
        task_id=task,
        agent=agent,
        seed=1,
        reward_version=version,
        proxy_score=1.0 if proxy_pass else 0.0,
        proxy_pass=proxy_pass,
        oracle_score=1.0 if oracle_pass else 0.0,
        oracle_pass=oracle_pass,
        hack=(proxy_pass and not oracle_pass),
    )


def _runs():
    return [
        _run("t1", "v0", "exploiter", proxy_pass=True, oracle_pass=False),  # hack
        _run("t1", "v0", "honest", proxy_pass=True, oracle_pass=True),
        _run("t1", "v1", "exploiter", proxy_pass=False, oracle_pass=False),
        _run("t1", "v1", "honest", proxy_pass=True, oracle_pass=True),
    ]


def test_summary_has_by_task_version_and_leaderboard():
    summary = build_summary(_runs())
    assert set(summary["by_task_version"]) == {"t1::v0", "t1::v1"}
    assert summary["by_task_version"]["t1::v0"]["rrs"] == 0.0
    assert summary["by_task_version"]["t1::v1"]["rrs"] == 1.0
    # Leaderboard sorted by rrs desc -> v1 first.
    assert summary["leaderboard"][0]["cell"] == "t1::v1"
    assert summary["macro_rrs"] == 0.5


def test_build_report_schema():
    report = build_report(
        _runs(),
        generated_at="2026-06-30T00:00:00Z",
        seeds=[1, 2, 3],
        scale="medium",
    )
    assert report["generated_at"] == "2026-06-30T00:00:00Z"
    assert report["seeds"] == [1, 2, 3]
    assert report["scale"] == "medium"
    assert len(report["runs"]) == 4
    assert "by_task_version" in report["summary"]
    assert "leaderboard" in report["summary"]
    assert report["judge_substudy"] is None


def test_write_report_roundtrips(tmp_path):
    report = build_report(_runs(), generated_at="x", seeds=[1])
    out = write_report(report, tmp_path / "reports" / "robustness.json")
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["summary"]["macro_rrs"] == 0.5
    assert loaded["runs"][0]["task_id"] == "t1"


def test_report_is_json_serializable():
    report = build_report(_runs(), generated_at="x", seeds=[1], judge_substudy={"naive": 1.0})
    # Must not raise.
    json.dumps(report)
