"""Unit tests for the statistical machinery behind model validation.

Added during a requirements audit, which found `models/validation/*` and
`models/optimization/*` at **0% coverage**. Those modules are Part 1
deliverables - the validation pipeline, A/B testing, drift detection and
regression gates - and they decide whether a model ships. Untested code that
decides whether code ships is the wrong thing to leave untested.

These target the pure functions: the statistics, the thresholds and the
guards. The CLI wrappers around them are exercised by
`tests/integration/test_model_loading.py` and by CI's `models` job.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from models.validation.ab_test import (
    IncomparableLabelSpacesError,
    TrafficSplit,
    assert_comparable_label_spaces,
    mcnemar_test,
    proportion_confidence_interval,
    verify_split_balance,
)
from models.validation.drift import (
    chi_square_drift,
    image_statistics,
    ks_drift,
    population_stability_index,
)
from models.validation.validate import expected_calibration_error, softmax


class TestSoftmax:
    """Turning raw scores into probabilities, without overflowing."""

    def test_sums_to_one(self) -> None:
        out = softmax(np.array([1.0, 2.0, 3.0]))
        assert out.sum() == pytest.approx(1.0)

    def test_preserves_ranking(self) -> None:
        out = softmax(np.array([0.5, 3.0, -1.0]))
        assert int(out.argmax()) == 1

    def test_survives_huge_logits(self) -> None:
        """exp(1000) is inf. The max-subtraction trick is what prevents nan."""
        out = softmax(np.array([1000.0, 999.0, 998.0]))
        assert np.all(np.isfinite(out))
        assert out.sum() == pytest.approx(1.0)
        assert int(out.argmax()) == 0

    def test_survives_hugely_negative_logits(self) -> None:
        out = softmax(np.array([-1000.0, -1001.0]))
        assert np.all(np.isfinite(out))
        assert out.sum() == pytest.approx(1.0)

    def test_uniform_logits_give_uniform_probabilities(self) -> None:
        out = softmax(np.zeros(4))
        assert out == pytest.approx(np.full(4, 0.25))

    def test_batched_rows_each_sum_to_one(self) -> None:
        out = softmax(np.array([[1.0, 2.0], [3.0, 0.0]]))
        assert out.sum(axis=-1) == pytest.approx([1.0, 1.0])


class TestMcNemarTest:
    """The paired test behind every A/B verdict."""

    def test_identical_models_are_a_tie(self) -> None:
        results = [True, False, True, True, False]
        _statistic, p_value, b, c = mcnemar_test(results, results)
        assert b == 0 and c == 0
        assert p_value == pytest.approx(1.0)

    def test_counts_only_disagreements(self) -> None:
        """Samples both models get right (or wrong) carry no information."""
        champion = [True, True, False, False]
        challenger = [True, False, True, False]
        _, _, b, c = mcnemar_test(champion, challenger)
        assert (b, c) == (1, 1)  # one each way; the two agreements ignored

    def test_clear_challenger_win_is_significant(self) -> None:
        champion = [False] * 30 + [True] * 5
        challenger = [True] * 30 + [True] * 5
        _, p_value, b, c = mcnemar_test(champion, challenger)
        assert c == 30 and b == 0
        assert p_value < 0.01

    def test_rejects_unequal_lengths(self) -> None:
        """A paired test on unpaired data is meaningless, not merely wrong."""
        with pytest.raises(ValueError):
            mcnemar_test([True, False], [True])


class TestProportionConfidenceInterval:
    def test_interval_brackets_the_difference(self) -> None:
        low, high = proportion_confidence_interval(80, 100, 90, 100)
        assert low < 0.10 < high

    def test_more_data_narrows_the_interval(self) -> None:
        narrow = proportion_confidence_interval(8000, 10000, 9000, 10000)
        wide = proportion_confidence_interval(8, 10, 9, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_no_difference_interval_contains_zero(self) -> None:
        low, high = proportion_confidence_interval(50, 100, 50, 100)
        assert low <= 0.0 <= high


class TestTrafficSplit:
    """Deterministic bucketing: the same user must always see the same model."""

    def test_same_user_always_gets_the_same_model(self) -> None:
        split = TrafficSplit(champion="a", challenger="b", challenger_percent=50.0)
        first = split.variant_for("user-123")
        assert all(split.variant_for("user-123") == first for _ in range(20))

    def test_split_is_roughly_the_requested_percentage(self) -> None:
        split = TrafficSplit(champion="a", challenger="b", challenger_percent=25.0)
        balance = verify_split_balance(split, sample_users=20_000)
        assert balance["actual_challenger_percent"] == pytest.approx(25.0, abs=1.5)

    def test_zero_percent_sends_everyone_to_the_champion(self) -> None:
        split = TrafficSplit(champion="a", challenger="b", challenger_percent=0.0)
        balance = verify_split_balance(split, sample_users=2_000)
        assert balance["actual_challenger_percent"] == pytest.approx(0.0)

    def test_disabled_split_sends_everyone_to_the_champion(self) -> None:
        split = TrafficSplit(champion="a", challenger="b", challenger_percent=50.0, enabled=False)
        assert {split.variant_for(f"u{i}") for i in range(50)} == {"a"}

    def test_describe_reports_the_configuration(self) -> None:
        split = TrafficSplit(champion="a", challenger="b", challenger_percent=10.0)
        described = split.describe()
        assert described["champion"] == "a"
        assert described["challenger"] == "b"
        assert described["challenger_percent"] == 10.0


class TestLabelSpaceGuard:
    """Refuses comparisons whose result could not mean anything.

    Regression test for an audit finding: comparing a 1000-class ImageNet model
    against a 200-class Tiny-ImageNet evaluation set reported
    "0% -> 86.67%, p=0.0000, promote the challenger". Arithmetically valid,
    completely meaningless.
    """

    def test_accepts_matching_label_spaces(self) -> None:
        assert_comparable_label_spaces({"a": 200, "b": 200}, dataset_classes=200)

    def test_rejects_model_with_far_more_classes_than_the_dataset(self) -> None:
        with pytest.raises(IncomparableLabelSpacesError, match="1000"):
            assert_comparable_label_spaces({"a": 1000, "b": 200}, dataset_classes=200)

    def test_rejects_dataset_with_more_classes_than_the_model(self) -> None:
        with pytest.raises(IncomparableLabelSpacesError):
            assert_comparable_label_spaces({"a": 10}, dataset_classes=200)

    def test_rejects_two_models_that_disagree_with_each_other(self) -> None:
        with pytest.raises(IncomparableLabelSpacesError, match="different label spaces"):
            assert_comparable_label_spaces({"a": 200, "b": 201}, dataset_classes=200)

    def test_unknown_class_counts_are_skipped_not_guessed(self) -> None:
        """A registry entry without num_classes should not block the run."""
        assert_comparable_label_spaces({"a": None, "b": None}, dataset_classes=200)


class TestKsDrift:
    """Kolmogorov-Smirnov: has a continuous distribution moved?"""

    def test_same_distribution_shows_no_drift(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 500).tolist()
        cur = rng.normal(0, 1, 500).tolist()
        result = ks_drift(ref, cur)
        assert not result.drifted

    def test_shifted_distribution_is_detected(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 500).tolist()
        cur = rng.normal(3, 1, 500).tolist()
        result = ks_drift(ref, cur)
        assert result.drifted
        assert result.p_value < 0.01

    def test_too_little_data_reports_rather_than_guesses(self) -> None:
        result = ks_drift([1.0, 2.0], [1.5, 2.5])
        assert not result.drifted


class TestChiSquareDrift:
    """Chi-square: has a categorical distribution moved?"""

    def test_same_label_mix_shows_no_drift(self) -> None:
        ref = ["cat"] * 100 + ["dog"] * 100
        cur = ["cat"] * 98 + ["dog"] * 102
        assert not chi_square_drift(ref, cur).drifted

    def test_inverted_label_mix_is_detected(self) -> None:
        ref = ["cat"] * 180 + ["dog"] * 20
        cur = ["cat"] * 20 + ["dog"] * 180
        result = chi_square_drift(ref, cur)
        assert result.drifted
        assert result.p_value < 0.01

    def test_a_brand_new_category_is_handled(self) -> None:
        """A label absent from the reference must not divide by zero."""
        result = chi_square_drift(["cat"] * 50, ["cat"] * 40 + ["fox"] * 10)
        assert isinstance(result.drifted, bool)


class TestPopulationStabilityIndex:
    """PSI: the industry convention for 'how far has this moved?'"""

    def test_identical_distributions_score_near_zero(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(0, 1, 1000).tolist()
        result = population_stability_index(values, values)
        assert result.statistic < 0.1
        assert not result.drifted

    def test_large_shift_exceeds_the_threshold(self) -> None:
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, 1000).tolist()
        cur = rng.normal(4, 1, 1000).tolist()
        result = population_stability_index(ref, cur)
        assert result.statistic > 0.25
        assert result.drifted


class TestImageStatistics:
    """Cheap per-image features drift detection can track over time."""

    @staticmethod
    def _png(colour: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", size, colour).save(buf, format="PNG")
        return buf.getvalue()

    def test_returns_finite_numbers(self) -> None:
        stats = image_statistics(self._png((128, 128, 128)))
        assert stats
        assert all(np.isfinite(v) for v in stats.values())

    def test_brighter_image_has_higher_mean(self) -> None:
        dark = image_statistics(self._png((10, 10, 10)))
        bright = image_statistics(self._png((240, 240, 240)))
        assert bright["mean_brightness"] > dark["mean_brightness"]

    def test_flat_image_has_almost_no_variation(self) -> None:
        stats = image_statistics(self._png((100, 100, 100)))
        assert stats["contrast"] == pytest.approx(0.0, abs=1e-6)


class TestExpectedCalibrationError:
    """Does a stated confidence of 0.8 mean it is right 80% of the time?"""

    def test_perfectly_calibrated_scores_near_zero(self) -> None:
        confidences = [0.9] * 100
        correct = [True] * 90 + [False] * 10
        ece, _ = expected_calibration_error(confidences, correct)
        assert ece == pytest.approx(0.0, abs=0.02)

    def test_overconfident_model_scores_high(self) -> None:
        """Claims 99% certainty, is right half the time."""
        confidences = [0.99] * 100
        correct = [True] * 50 + [False] * 50
        ece, _ = expected_calibration_error(confidences, correct)
        assert ece > 0.4

    def test_returns_one_entry_per_populated_bin(self) -> None:
        confidences = [0.15, 0.45, 0.85]
        correct = [False, True, True]
        _, bins = expected_calibration_error(confidences, correct, bins=10)
        assert 1 <= len(bins) <= 10

    def test_empty_input_does_not_explode(self) -> None:
        ece, bins = expected_calibration_error([], [])
        assert ece == 0.0
        assert bins == []
