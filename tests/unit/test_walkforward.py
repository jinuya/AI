"""Walk-forward and purged k-fold splits — spec §8.4."""

from __future__ import annotations

from decimal import Decimal

import pytest

from atrader.backtest.walkforward import purged_kfold_splits, walk_forward_splits


class TestWalkForwardSplits:
    def test_rolling_windows_never_train_on_the_future(self) -> None:
        splits = walk_forward_splits(100, train_size=20, test_size=10)
        for split in splits:
            assert max(split.train_indices) < min(split.test_indices)

    def test_rolling_windows_slide_forward_by_step(self) -> None:
        splits = walk_forward_splits(100, train_size=20, test_size=10)
        assert splits[0].train_indices == tuple(range(0, 20))
        assert splits[0].test_indices == tuple(range(20, 30))
        assert splits[1].train_indices == tuple(range(10, 30))  # step defaults to test_size
        assert splits[1].test_indices == tuple(range(30, 40))

    def test_anchored_windows_always_start_at_zero(self) -> None:
        splits = walk_forward_splits(100, train_size=20, test_size=10, anchored=True)
        assert splits[0].train_indices[0] == 0
        assert splits[1].train_indices[0] == 0
        assert len(splits[1].train_indices) > len(splits[0].train_indices)

    def test_a_custom_step_is_respected(self) -> None:
        splits = walk_forward_splits(100, train_size=20, test_size=10, step=5)
        assert splits[1].test_indices[0] == splits[0].test_indices[0] + 5

    def test_stops_once_a_full_window_no_longer_fits(self) -> None:
        splits = walk_forward_splits(35, train_size=20, test_size=10)
        assert len(splits) == 1  # a second window would need index 45, beyond 35

    def test_rejects_non_positive_train_or_test_size(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            walk_forward_splits(100, train_size=0, test_size=10)
        with pytest.raises(ValueError, match="must be positive"):
            walk_forward_splits(100, train_size=10, test_size=0)

    def test_rejects_a_non_positive_step(self) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            walk_forward_splits(100, train_size=10, test_size=10, step=0)


class TestPurgedKFoldSplits:
    def test_every_sample_appears_in_exactly_one_test_fold(self) -> None:
        splits = purged_kfold_splits(100, n_splits=5)
        all_test = sorted(i for split in splits for i in split.test_indices)
        assert all_test == list(range(100))

    def test_without_a_label_span_or_embargo_training_is_everything_but_the_test_fold(
        self,
    ) -> None:
        splits = purged_kfold_splits(100, n_splits=5, label_span=1, embargo_pct=Decimal("0"))
        first = splits[0]
        assert set(first.train_indices) == set(range(100)) - set(first.test_indices)

    def test_a_label_span_purges_training_samples_just_before_the_test_fold(self) -> None:
        splits = purged_kfold_splits(100, n_splits=5, label_span=5)
        # Fold 1's test fold starts at 20; a 5-step label span means indices
        # 16..19 have labels overlapping the test fold and must be purged.
        fold_1 = splits[1]
        assert fold_1.test_indices[0] == 20
        for i in range(16, 20):
            assert i not in fold_1.train_indices

    def test_embargo_purges_training_samples_just_after_the_test_fold(self) -> None:
        splits = purged_kfold_splits(100, n_splits=5, embargo_pct=Decimal("10"))
        fold_0 = splits[0]
        assert fold_0.test_indices[-1] == 19
        # embargo = 10% of 100 = 10 samples immediately after the test fold.
        for i in range(20, 30):
            assert i not in fold_0.train_indices
        assert 30 in fold_0.train_indices

    def test_the_last_fold_absorbs_any_remainder(self) -> None:
        splits = purged_kfold_splits(103, n_splits=5)
        assert splits[-1].test_indices[-1] == 102

    def test_rejects_fewer_than_two_splits(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            purged_kfold_splits(100, n_splits=1)

    def test_rejects_a_label_span_below_one(self) -> None:
        with pytest.raises(ValueError, match="label_span"):
            purged_kfold_splits(100, n_splits=5, label_span=0)

    def test_rejects_an_out_of_range_embargo(self) -> None:
        with pytest.raises(ValueError, match="embargo_pct"):
            purged_kfold_splits(100, n_splits=5, embargo_pct=Decimal("150"))
