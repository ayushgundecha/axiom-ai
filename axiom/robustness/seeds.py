"""Held-out seed split (RR20).

Robustness must generalize, not overfit. Hardening decisions (the v1 proxy
specs) are made against a set of **train** seeds; the reported RRS is computed on
a DISJOINT set of **eval** seeds. If a defense only works on seeds it was tuned
on, the held-out RRS exposes it.

These are tiny pure helpers; the harness uses them to validate
``--train-seeds`` vs ``--eval-seeds`` and to record the split in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

from axiom.exceptions import TaskConfigError


@dataclass(frozen=True)
class SeedSplit:
    """A train/eval seed partition."""

    train: tuple[int, ...]
    eval: tuple[int, ...]

    @property
    def held_out(self) -> bool:
        """True when eval seeds are fully disjoint from train seeds."""
        return bool(self.eval) and not (set(self.train) & set(self.eval))

    def to_dict(self) -> dict[str, list[int] | bool]:
        return {
            "train_seeds": list(self.train),
            "eval_seeds": list(self.eval),
            "held_out": self.held_out,
        }


def validate_split(train: list[int], eval_: list[int], *, strict: bool = False) -> None:
    """Validate a train/eval split.

    Raises if either side is empty, or (when ``strict``) if they overlap — a
    leaked seed would make the reported RRS optimistic.
    """
    if not train:
        raise TaskConfigError("train seeds must be non-empty")
    if not eval_:
        raise TaskConfigError("eval seeds must be non-empty")
    overlap = set(train) & set(eval_)
    if overlap and strict:
        raise TaskConfigError(f"train/eval seeds overlap (leakage): {sorted(overlap)}")


def make_split(train: list[int], eval_: list[int] | None) -> SeedSplit:
    """Build a :class:`SeedSplit`; eval defaults to train (not held out)."""
    ev = eval_ if eval_ is not None else list(train)
    return SeedSplit(train=tuple(train), eval=tuple(ev))


def auto_split(all_seeds: list[int], n_train: int) -> SeedSplit:
    """Split a seed list into the first ``n_train`` (train) and the rest (eval)."""
    if n_train < 1 or n_train >= len(all_seeds):
        raise TaskConfigError(
            f"n_train must be in [1, {len(all_seeds) - 1}] for {len(all_seeds)} seeds"
        )
    return SeedSplit(train=tuple(all_seeds[:n_train]), eval=tuple(all_seeds[n_train:]))
