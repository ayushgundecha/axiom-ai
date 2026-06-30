"""Held-out seed split (RR20)."""

from __future__ import annotations

import pytest

from axiom.exceptions import TaskConfigError
from axiom.robustness.seeds import auto_split, make_split, validate_split


def test_held_out_split_is_disjoint():
    split = make_split([1, 2, 3], [4, 5])
    assert split.held_out is True
    d = split.to_dict()
    assert d == {"train_seeds": [1, 2, 3], "eval_seeds": [4, 5], "held_out": True}


def test_eval_defaults_to_train_not_held_out():
    split = make_split([1, 2, 3], None)
    assert split.eval == (1, 2, 3)
    assert split.held_out is False


def test_overlap_not_held_out():
    assert make_split([1, 2, 3], [3, 4]).held_out is False


def test_validate_split_strict_rejects_overlap():
    with pytest.raises(TaskConfigError, match="overlap"):
        validate_split([1, 2], [2, 3], strict=True)
    # non-strict tolerates overlap
    validate_split([1, 2], [2, 3], strict=False)


def test_validate_split_requires_nonempty():
    with pytest.raises(TaskConfigError, match="train"):
        validate_split([], [1])
    with pytest.raises(TaskConfigError, match="eval"):
        validate_split([1], [])


def test_auto_split():
    split = auto_split([1, 2, 3, 4, 5], n_train=3)
    assert split.train == (1, 2, 3)
    assert split.eval == (4, 5)
    assert split.held_out is True
    with pytest.raises(TaskConfigError):
        auto_split([1, 2], n_train=2)
