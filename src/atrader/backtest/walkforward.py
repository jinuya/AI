"""Walk-forward and purged k-fold splits — spec §8.4.

    워크포워드 + purged K-fold + embargo.

Time-series data breaks the IID assumption ordinary cross-validation relies
on. Two related leaks show up if that assumption is applied anyway:

* **Look-ahead.** A fold's training set contains observations from *after*
  its test set. A model "predicting" the past using the future is not
  predicting anything — :func:`walk_forward_splits` avoids this structurally
  by only ever training on indices that precede the test window.
* **Leakage through label overlap.** Even a train/test split that does not
  cross in calendar time can still leak if a sample's *label* — often built
  from a forward-looking window, e.g. "return over the next 5 days" — spans
  the train/test boundary. A training sample sitting one bar before the test
  fold, whose label was computed from bars that fall inside the test fold, has
  effectively seen the test fold already.

:func:`purged_kfold_splits` removes training samples whose label window
overlaps the test fold (**purging**), and additionally removes a further
margin of samples immediately after the test fold (**embargo**), because a
fold's influence through serial correlation does not end exactly at its last
observation — the embargo is the difference between "definitely no overlap"
and "no overlap or anything close to it".
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

__all__ = ["Split", "purged_kfold_splits", "walk_forward_splits"]


@dataclass(frozen=True, slots=True)
class Split:
    """Indices into a chronologically ordered dataset (e.g. daily bars)."""

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def walk_forward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    anchored: bool = False,
) -> list[Split]:
    """Rolling (or anchored) walk-forward windows.

    Rolling: the training window slides forward with the test window, always
    ``train_size`` long. Anchored: the training window always starts at index
    0 and grows — appropriate when more history is strictly better for the
    model, at the cost of later folds training on much more data than
    earlier ones (and therefore not being directly comparable to them).
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step if step is not None else test_size
    if step <= 0:
        raise ValueError("step must be positive")

    splits: list[Split] = []
    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n_samples:
            break
        train_start = 0 if anchored else start
        splits.append(
            Split(
                train_indices=tuple(range(train_start, train_end)),
                test_indices=tuple(range(train_end, test_end)),
            )
        )
        start += step
    return splits


def purged_kfold_splits(
    n_samples: int,
    *,
    n_splits: int,
    embargo_pct: Decimal = Decimal("0"),
    label_span: int = 1,
) -> list[Split]:
    """K-fold splits with the test fold's neighborhood purged from training.

    ``label_span`` is how many samples ahead a label looks (1 means "no
    forward-looking label, purge nothing extra on that account"; 5 means
    "each sample's label depends on the next 5 observations"). ``embargo_pct``
    is an *additional* margin after the test fold, as a percentage of the
    total dataset — spec-recommended because label overlap alone
    understates how far a fold's influence actually reaches.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if label_span < 1:
        raise ValueError(f"label_span must be at least 1, got {label_span}")
    if not 0 <= embargo_pct <= 100:
        raise ValueError(f"embargo_pct must be within 0..100, got {embargo_pct}")

    fold_size = n_samples // n_splits
    embargo = int(n_samples * embargo_pct / Decimal(100))

    splits: list[Split] = []
    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = n_samples if fold == n_splits - 1 else test_start + fold_size
        purge_start = max(0, test_start - (label_span - 1))
        purge_end = min(n_samples, test_end + embargo)

        train_indices = tuple(i for i in range(n_samples) if i < purge_start or i >= purge_end)
        splits.append(
            Split(train_indices=train_indices, test_indices=tuple(range(test_start, test_end)))
        )
    return splits
