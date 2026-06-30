"""Smoke test for the offline robustness harness (RR13)."""

from __future__ import annotations

import importlib.util
from datetime import UTC
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "run_robustness.py"
_TASKS_DIR = _ROOT / "tasks"


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_robustness", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HARNESS = _load_harness()


@pytest.mark.asyncio
async def test_offline_harness_shows_v0_gameable_v1_robust():
    tasks = HARNESS._load_tasks(_TASKS_DIR, HARNESS.ALL_TASKS)
    cases = HARNESS.load_corpus()
    labels = await HARNESS.run_offline(tasks, cases, seeds=[1, 2], versions=["v0", "v1"])
    assert labels

    by = HARNESS.rrs_by_task_version(labels)
    v0 = [s for k, s in by.items() if k.endswith("::v0")]
    v1 = [s for k, s in by.items() if k.endswith("::v1")]

    # Naive proxies are gameable somewhere; hardened proxies are not, anywhere.
    assert any(s.hack_rate > 0 for s in v0)
    assert all(s.hack_rate == 0.0 for s in v1)
    assert all(s.honest_fidelity == 1.0 for s in v1)
    assert HARNESS.macro_rrs(v1) == 1.0
    assert HARNESS.macro_rrs(v0) < 1.0


@pytest.mark.asyncio
async def test_offline_harness_writes_report(tmp_path):
    tasks = HARNESS._load_tasks(_TASKS_DIR, ["answer_support_question"])
    cases = [c for c in HARNESS.load_corpus() if c.task_id == "answer_support_question"]
    labels = await HARNESS.run_offline(tasks, cases, seeds=[1], versions=["v0", "v1"])

    from datetime import datetime

    report = HARNESS.build_report(labels, generated_at=datetime.now(UTC).isoformat(), seeds=[1])
    out = HARNESS.write_report(report, tmp_path / "robustness.json")
    assert out.exists()
    assert report["summary"]["by_task_version"]
