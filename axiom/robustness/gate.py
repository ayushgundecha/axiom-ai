"""CI gate + reporting-only robustness evaluator (RR25).

Two things keep the robustness work honest in CI and in code:

  * **CI gate** — fail the build if the hardened (v1) reward regresses: macro-RRS
    on the eval seeds drops below a threshold, or any exploiter run becomes a
    hack again. This turns "rewards you can't cheat" into an enforced invariant.
  * **Reporting-only evaluator** — the oracle / proxy-oracle gap is exposed for
    dashboards and the RRS, but is structurally prevented from ever becoming the
    live training reward (the load-bearing invariant). Gated behind
    ``settings.robustness_reporting_only`` and a guard that raises if anyone
    tries to use the oracle signal as a live reward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from axiom.config import AxiomSettings, get_settings
from axiom.exceptions import EvaluationError
from axiom.robustness.labeler import RunLabel
from axiom.robustness.metrics import compute_rrs, rrs_by_task_version


@dataclass(frozen=True)
class GateResult:
    """Outcome of the CI robustness gate."""

    passed: bool
    macro_rrs: float
    hack_rate: float
    reasons: list[str] = field(default_factory=list)


def run_ci_gate(
    labeled_runs: Sequence[RunLabel],
    *,
    version: str = "v1",
    min_macro_rrs: float | None = None,
    max_hack_rate: float | None = None,
    use_gap_variance: bool = False,
) -> GateResult:
    """Gate the hardened reward version: macro-RRS high enough, hack_rate low enough."""
    settings = get_settings()
    min_rrs = settings.robustness_min_macro_rrs if min_macro_rrs is None else min_macro_rrs
    max_hr = settings.robustness_max_hack_rate if max_hack_rate is None else max_hack_rate

    version_runs = [r for r in labeled_runs if r.reward_version == version]
    if not version_runs:
        return GateResult(False, 0.0, 0.0, [f"no runs for reward version {version!r}"])

    overall = compute_rrs(version_runs, use_gap_variance=use_gap_variance)
    by = rrs_by_task_version(version_runs, use_gap_variance=use_gap_variance)
    macro = round(sum(s.rrs for s in by.values()) / len(by), 4) if by else 0.0

    reasons: list[str] = []
    if macro < min_rrs:
        reasons.append(f"macro-RRS {macro} < threshold {min_rrs}")
    if overall.hack_rate > max_hr:
        reasons.append(f"hack_rate {overall.hack_rate} > threshold {max_hr}")
    for cell, score in by.items():
        if score.hack_rate > max_hr:
            reasons.append(f"{cell}: hack_rate {score.hack_rate} > {max_hr}")

    return GateResult(
        passed=not reasons, macro_rrs=macro, hack_rate=overall.hack_rate, reasons=reasons
    )


class ReportingOnlyRobustness:
    """Surfaces the robustness signal for reporting — never as a live reward.

    Construct via :func:`reporting_evaluator`, which enforces the
    ``robustness_reporting_only`` flag. Any attempt to coerce the oracle signal
    into a live reward raises, keeping the load-bearing invariant true in code.
    """

    reporting_only = True

    def __init__(self, settings: AxiomSettings) -> None:
        if not settings.robustness_reporting_only:
            msg = (
                "robustness_reporting_only must be True — the oracle signal is a "
                "referee, never the live training reward"
            )
            raise EvaluationError(msg)
        self._settings = settings

    def gap(self, label: RunLabel) -> float:
        """The proxy − oracle gap for a labeled run (reporting metric)."""
        return round(label.proxy_score - label.oracle_score, 4)

    def is_hack(self, label: RunLabel) -> bool:
        return label.hack

    def as_live_reward(self, label: RunLabel) -> float:
        """Forbidden: the oracle signal must never be the live reward."""
        msg = "the oracle/robustness signal is reporting-only and cannot be a live reward"
        raise EvaluationError(msg)


def reporting_evaluator(settings: AxiomSettings | None = None) -> ReportingOnlyRobustness:
    """Build the reporting-only evaluator (raises unless the flag is set)."""
    return ReportingOnlyRobustness(settings or get_settings())
