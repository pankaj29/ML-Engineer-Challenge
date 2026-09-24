"""Unit tests for the validation checks and the regression gate.

Part of the same audit finding as `test_validation_statistics.py`:
`models/validation/*` shipped at 0% coverage despite being the machinery that
decides whether a model is allowed to ship.

These use small fake `infer` callables rather than real models, so each check
can be shown to pass on good behaviour **and fail on the specific defect it
exists to catch** - which is the part that matters. A check that never fails
in testing is not a check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from models.validation.regression import (
    BaselineStore,
    check_metric,
    check_regression,
    environment_fingerprint,
)
from models.validation.validate import (
    check_artifacts,
    check_batch_invariance,
    check_determinism,
    check_output_sanity,
    check_robustness,
)

SAMPLE = np.zeros((1, 3, 8, 8), dtype=np.float32)


def _probs(n: int = 4, batch: int = 1) -> list[np.ndarray]:
    """A well-behaved probability output."""
    row = np.full(n, 1.0 / n, dtype=np.float32)
    return [np.tile(row, (batch, 1))]


class TestCheckDeterminism:
    def test_passes_when_output_never_changes(self) -> None:
        result = check_determinism(lambda x: _probs(batch=x.shape[0]), SAMPLE)
        assert result.passed

    def test_fails_when_output_wanders(self) -> None:
        """Catches nondeterminism from unseeded dropout or thread races."""
        rng = np.random.default_rng(0)

        def unstable(x: np.ndarray) -> list[np.ndarray]:
            return [rng.random((x.shape[0], 4)).astype(np.float32)]

        assert not check_determinism(unstable, SAMPLE).passed


class TestCheckBatchInvariance:
    def test_passes_when_batching_changes_nothing(self) -> None:
        result = check_batch_invariance(lambda x: _probs(batch=x.shape[0]), SAMPLE)
        assert result.passed

    def test_fails_when_a_prediction_depends_on_its_neighbours(self) -> None:
        """The defect: an image scored differently alone vs in a batch."""

        def batch_sensitive(x: np.ndarray) -> list[np.ndarray]:
            out = np.tile(np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32), (x.shape[0], 1))
            if x.shape[0] > 1:
                out[0] = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
            return [out]

        assert not check_batch_invariance(batch_sensitive, SAMPLE).passed


class TestCheckOutputSanity:
    def test_passes_on_a_valid_probability_vector(self) -> None:
        assert check_output_sanity(lambda x: _probs(batch=x.shape[0]), SAMPLE).passed

    def test_fails_on_nan(self) -> None:
        def nan_out(x: np.ndarray) -> list[np.ndarray]:
            return [np.full((x.shape[0], 4), np.nan, dtype=np.float32)]

        assert not check_output_sanity(nan_out, SAMPLE).passed

    def test_fails_on_infinity(self) -> None:
        def inf_out(x: np.ndarray) -> list[np.ndarray]:
            return [np.full((x.shape[0], 4), np.inf, dtype=np.float32)]

        assert not check_output_sanity(inf_out, SAMPLE).passed

    def test_fails_when_probabilities_do_not_sum_to_one(self) -> None:
        """Values in [0, 1] that do not sum to 1 are a broken distribution.

        Regression test: this branch used to softmax the output before
        checking the sum, and softmax always sums to 1 - so the check could
        never fail. It now inspects the raw output.
        """

        def broken_distribution(x: np.ndarray) -> list[np.ndarray]:
            # In [0, 1], so it claims to be probabilities - but sums to 2.0.
            return [np.full((x.shape[0], 4), 0.5, dtype=np.float32)]

        assert not check_output_sanity(
            broken_distribution, SAMPLE, expect_probabilities=True
        ).passed

    def test_raw_logits_from_a_classifier_are_accepted(self) -> None:
        """Our models emit logits, not probabilities. That is not a defect."""

        def logit_out(x: np.ndarray) -> list[np.ndarray]:
            return [np.array([[8.2, 3.1, -1.4, 0.5]] * x.shape[0], dtype=np.float32)]

        assert check_output_sanity(logit_out, SAMPLE, expect_probabilities=True).passed

    def test_raw_logits_allowed_when_not_expecting_probabilities(self) -> None:
        def logits(x: np.ndarray) -> list[np.ndarray]:
            return [np.array([[4.0, -2.0, 0.5, 1.0]] * x.shape[0], dtype=np.float32)]

        assert check_output_sanity(logits, SAMPLE, expect_probabilities=False).passed


class TestCheckRobustness:
    def test_passes_when_tiny_noise_changes_nothing(self) -> None:
        def stable(x: np.ndarray) -> list[np.ndarray]:
            return [np.tile(np.array([0.9, 0.1], dtype=np.float32), (x.shape[0], 1))]

        samples = [np.zeros((1, 3, 8, 8), dtype=np.float32) for _ in range(5)]
        assert check_robustness(stable, samples).passed

    def test_fails_when_imperceptible_noise_flips_predictions(self) -> None:
        """A model this brittle is unfit to serve, however good its accuracy."""
        flip = {"n": 0}

        def brittle(x: np.ndarray) -> list[np.ndarray]:
            flip["n"] += 1
            row = [0.9, 0.1] if flip["n"] % 2 else [0.1, 0.9]
            return [np.tile(np.array(row, dtype=np.float32), (x.shape[0], 1))]

        samples = [np.zeros((1, 3, 8, 8), dtype=np.float32) for _ in range(6)]
        assert not check_robustness(brittle, samples).passed


class TestCheckArtifacts:
    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        class Entry:
            key = "m:1.0.0"
            artifacts = {"onnx": "nope.onnx"}
            labels_file = None

        assert not check_artifacts(Entry(), tmp_path).passed

    def test_passes_when_every_artifact_is_present(self, tmp_path: Path) -> None:
        (tmp_path / "m.onnx").write_bytes(b"x" * 64)

        class Entry:
            key = "m:1.0.0"
            artifacts = {"onnx": "m.onnx"}
            labels_file = None

        assert check_artifacts(Entry(), tmp_path).passed


class TestCheckMetric:
    """Direction matters: lower is better for latency, higher for throughput."""

    def test_latency_improvement_passes(self) -> None:
        assert check_metric("p95_latency_ms", baseline=100.0, current=80.0).passed

    def test_latency_regression_fails(self) -> None:
        assert not check_metric("p95_latency_ms", baseline=100.0, current=200.0).passed

    def test_throughput_improvement_passes(self) -> None:
        assert check_metric("throughput_ips", baseline=10.0, current=20.0).passed

    def test_throughput_collapse_fails(self) -> None:
        assert not check_metric("throughput_ips", baseline=20.0, current=5.0).passed

    def test_small_movement_is_within_tolerance(self) -> None:
        """Run-to-run noise must not be reported as a regression."""
        assert check_metric("p95_latency_ms", baseline=100.0, current=102.0).passed

    def test_explicit_tolerance_is_honoured(self) -> None:
        assert not check_metric(
            "p95_latency_ms", baseline=100.0, current=105.0, tolerance=0.01
        ).passed


class TestBaselineStoreAndRegression:
    def test_records_then_reads_back(self, tmp_path: Path) -> None:
        store = BaselineStore(path=tmp_path / "baselines.json")
        store.record("m:1.0.0", {"p95_latency_ms": 100.0, "throughput_ips": 10.0})
        assert store.get("m:1.0.0") is not None

    def test_no_baseline_is_not_a_failure(self, tmp_path: Path) -> None:
        """First run of a new model has nothing to compare against."""
        store = BaselineStore(path=tmp_path / "baselines.json")
        report = check_regression("brand-new:1.0.0", {"p95_latency_ms": 50.0}, store=store)
        assert report.passed

    def test_detects_a_real_regression(self, tmp_path: Path) -> None:
        store = BaselineStore(path=tmp_path / "baselines.json")
        store.record("m:1.0.0", {"p95_latency_ms": 100.0})
        report = check_regression("m:1.0.0", {"p95_latency_ms": 300.0}, store=store)
        assert not report.passed

    def test_passes_when_performance_improves(self, tmp_path: Path) -> None:
        store = BaselineStore(path=tmp_path / "baselines.json")
        store.record("m:1.0.0", {"p95_latency_ms": 100.0})
        report = check_regression("m:1.0.0", {"p95_latency_ms": 60.0}, store=store)
        assert report.passed


class TestEnvironmentFingerprint:
    def test_records_enough_to_explain_a_number(self) -> None:
        """A benchmark without its environment is not reproducible."""
        fp = environment_fingerprint()
        assert fp
        blob = " ".join(str(v).lower() for v in fp.values()) + " ".join(fp)
        assert "python" in blob or "platform" in blob or "cpu" in blob

    def test_is_json_serialisable(self) -> None:
        import json

        json.dumps(environment_fingerprint())
