"""Judge sub-study: naive fooled, hardened resists, honest controls pass (RR19)."""

from __future__ import annotations

from axiom.robustness.judge_substudy import ATTACKS, run_judge_substudy


async def test_substudy_naive_fooled_hardened_resists():
    result = await run_judge_substudy()
    assert result["backend"] == "simulated"
    assert len(result["attacks"]) == len(ATTACKS) == 3

    for a in result["attacks"]:
        assert a["naive_fooled"] is True, f"{a['id']} should fool the naive judge"
        assert a["hardened_resisted"] is True, f"{a['id']} should be resisted by the hardened judge"

    assert result["summary"]["naive_fooled_rate"] == 1.0
    assert result["summary"]["hardened_resisted_rate"] == 1.0


async def test_substudy_honest_control_scores_well_on_both():
    result = await run_judge_substudy()
    control = result["honest_control"]
    assert control["naive_score"] >= 0.7
    assert control["hardened_score"] >= 0.7


async def test_substudy_is_json_serializable():
    import json

    result = await run_judge_substudy()
    json.dumps(result)  # must not raise
