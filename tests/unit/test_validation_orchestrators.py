"""End-to-end tests for the validation orchestrators and their CLIs.

`models/validation/*` sat at 54% because everything tested so far was a pure
statistical helper. The parts that were untested are the parts a release
actually depends on: `validate_model`, `measure_model`, `detect_drift`,
`compare`, and the four `main()` functions that CI invokes.

These run for real rather than against mocks. A throwaway registry is built in
`tmp_path`, pointing at a genuinely exported two-class ONNX model, and
`MODEL_REGISTRY_PATH` / `MODEL_ARTIFACTS_DIR` are redirected at it. So the
model really loads, really infers, and the checks really measure something -
which is the only way these tests can catch a broken check, as opposed to
confirming that a mock returns what the mock was told to return.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.validation import (
    ab_test as ab_mod,
    drift as drift_mod,
    regression as regression_mod,
    validate as validate_mod,
)
from models.validation.ab_test import ModelScores, compare, evaluate_models
from models.validation.drift import detect_drift
from models.validation.regression import BaselineStore, measure_model
from models.validation.validate import validate_model

torch = pytest.importorskip("torch")

NUM_CLASSES = 4
IMAGE_SIZE = 32


def _png(seed: int, size: int = IMAGE_SIZE) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> str:
    """A one-model registry backed by a real ONNX artifact.

    Deterministic by construction: the exported graph is a fixed-seed linear
    model, so `check_determinism` has something honest to verify.
    """
    from models.optimization.export_onnx import export_to_onnx

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES),
    ).eval()
    # Widen the output range on purpose. A default-initialised linear layer on
    # normalised input can land entirely inside [0, 1], and `check_output_sanity`
    # then - correctly - reports it as a probability vector that does not sum to
    # 1. That is the check working; it just makes the fixture flaky.
    with torch.no_grad():
        model[1].weight.mul_(50.0)
    export_to_onnx(
        model,
        artifacts / "tiny.onnx",
        input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        name="tiny",
    )

    (artifacts / "tiny_labels.json").write_text(
        json.dumps([f"class_{i}" for i in range(NUM_CLASSES)]), encoding="utf-8"
    )

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "models": [
                    {
                        "name": "tiny",
                        "version": "1.0.0",
                        "task": "classification",
                        "artifacts": {"onnx": "tiny.onnx"},
                        "preprocess": "imagenet_224",
                        "labels_file": "tiny_labels.json",
                        "num_classes": NUM_CLASSES,
                        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
                        "metrics": {},
                        "limitations": ["Test fixture; predicts nothing meaningful."],
                        "description": "Throwaway model for validation tests.",
                        "is_default": True,
                        "status": "active",
                    },
                    {
                        "name": "tiny-challenger",
                        "version": "1.0.0",
                        "task": "classification",
                        "artifacts": {"onnx": "tiny.onnx"},
                        "preprocess": "imagenet_224",
                        "labels_file": "tiny_labels.json",
                        "num_classes": NUM_CLASSES,
                        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
                        "metrics": {},
                        "limitations": ["Test fixture."],
                        "description": "Challenger fixture.",
                        "is_default": False,
                        "status": "active",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # `ModelService()` defaults to the module-level `settings` object, which
    # was built at import time - clearing the get_settings cache is not enough
    # to redirect it, so the module attribute itself is replaced.
    import api.services.model_service as model_service
    from api.config import Settings

    monkeypatch.setattr(
        model_service,
        "settings",
        Settings(model_registry_path=registry_path, model_artifacts_dir=artifacts),
    )
    return "tiny:1.0.0"


@pytest.fixture
def eval_samples() -> list[tuple[bytes, int]]:
    return [(_png(i), i % NUM_CLASSES) for i in range(12)]


def _run(module, argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr("sys.argv", [module.__name__, *argv])
    return module.main()


class TestValidateModel:
    def test_a_registered_model_passes_the_suite(self, registry: str) -> None:
        report = validate_model(registry, max_p95_ms=10_000.0)
        assert report.passed is True, report.critical_failures
        assert report.checks

    def test_every_documented_check_actually_runs(self, registry: str, eval_samples) -> None:
        report = validate_model(registry, eval_samples=eval_samples, max_p95_ms=10_000.0)
        names = {c["name"] for c in report.checks}
        for required in (
            "artifact_integrity",
            "determinism",
            "batch_invariance",
            "output_sanity",
            "latency",
            "accuracy",
            "calibration",
        ):
            assert required in names, f"{required} check did not run"

    def test_labelled_samples_unlock_accuracy_and_calibration(
        self, registry: str, eval_samples
    ) -> None:
        with_labels = validate_model(registry, eval_samples=eval_samples, max_p95_ms=10_000.0)
        without = validate_model(registry, max_p95_ms=10_000.0)
        assert len(with_labels.checks) > len(without.checks)

    def test_an_impossible_latency_budget_fails_the_model(self, registry: str) -> None:
        """Proves the budget is enforced, not just reported."""
        report = validate_model(registry, max_p95_ms=0.0)
        latency = next(c for c in report.checks if c["name"] == "latency")
        assert latency["passed"] is False

    def test_an_unregistered_model_fails_rather_than_raising(self, registry: str) -> None:
        report = validate_model("no-such-model:9.9.9")
        assert report.passed is False
        assert "not in the registry" in report.critical_failures[0]

    def test_the_report_serialises(self, registry: str) -> None:
        payload = validate_model(registry, max_p95_ms=10_000.0).to_dict()
        json.dumps(payload)
        assert payload["model"] == registry

    def test_summary_is_human_readable(self, registry: str) -> None:
        assert "tiny" in validate_model(registry, max_p95_ms=10_000.0).summary()


class TestValidateCli:
    def test_validates_every_registered_model(
        self, registry: str, tmp_path: Path, monkeypatch
    ) -> None:
        out = tmp_path / "validation.json"
        code = _run(
            validate_mod,
            ["--max-p95-ms", "10000", "--output", str(out)],
            monkeypatch,
        )
        assert code == 0
        assert json.loads(out.read_text(encoding="utf-8"))[0]["model"] == "tiny:1.0.0"

    def test_a_failing_model_gives_a_non_zero_exit_code(
        self, registry: str, tmp_path: Path, monkeypatch
    ) -> None:
        """CI relies on this: a validation failure must break the build."""
        code = _run(
            validate_mod,
            ["--model", "tiny:1.0.0", "--max-p95-ms", "0", "--output", str(tmp_path / "v.json")],
            monkeypatch,
        )
        assert code == 1

    def test_missing_eval_data_is_a_warning_not_a_crash(
        self, registry: str, tmp_path: Path, monkeypatch
    ) -> None:
        code = _run(
            validate_mod,
            [
                "--eval-samples",
                "10",
                "--data-dir",
                str(tmp_path / "no-data"),
                "--max-p95-ms",
                "10000",
                "--output",
                str(tmp_path / "v.json"),
            ],
            monkeypatch,
        )
        assert code == 0


class TestMeasureModel:
    def test_returns_the_latency_metrics_the_baseline_tracks(self, registry: str) -> None:
        metrics = measure_model(registry, iterations=5, warmup=1)
        for key in ("p50_latency_ms", "p95_latency_ms", "throughput_ips", "size_mb"):
            assert key in metrics
            assert metrics[key] > 0

    def test_an_unregistered_model_raises(self, registry: str) -> None:
        with pytest.raises(ValueError, match="not in the registry"):
            measure_model("absent:1.0.0", iterations=2, warmup=1)


class TestBaselineStore:
    def test_recording_then_reading_back(self, tmp_path: Path) -> None:
        store = BaselineStore(tmp_path / "baseline.json")
        store.record("tiny:1.0.0", {"p95_latency_ms": 5.0}, note="first")
        assert (
            BaselineStore(tmp_path / "baseline.json").get("tiny:1.0.0")["metrics"]["p95_latency_ms"]
            == 5.0
        )

    def test_an_unknown_model_has_no_baseline(self, tmp_path: Path) -> None:
        assert BaselineStore(tmp_path / "baseline.json").get("nothing") is None

    def test_a_corrupt_baseline_file_fails_loudly(self, tmp_path: Path) -> None:
        """Deliberate: silently discarding the baseline would make every
        later regression check pass vacuously, which is the one outcome a
        regression gate must never produce."""
        path = tmp_path / "baseline.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            BaselineStore(path)


class TestRegressionCli:
    """Three subcommands: `record`, `check`, `check-all`."""

    def test_record_writes_a_baseline(self, registry: str, tmp_path: Path, monkeypatch) -> None:
        baseline = tmp_path / "baseline.json"
        code = _run(
            regression_mod,
            [
                "record",
                "--model",
                "tiny:1.0.0",
                "--iterations",
                "5",
                "--baseline-file",
                str(baseline),
                "--note",
                "first run",
            ],
            monkeypatch,
        )
        assert code == 0
        stored = json.loads(baseline.read_text(encoding="utf-8"))["baselines"]["tiny:1.0.0"]
        assert stored["metrics"]["p95_latency_ms"] > 0
        assert stored["note"] == "first run"

    def test_check_against_a_recorded_baseline(
        self, registry: str, tmp_path: Path, monkeypatch
    ) -> None:
        baseline = tmp_path / "baseline.json"
        report = tmp_path / "regression.json"
        _run(
            regression_mod,
            [
                "record",
                "--model",
                "tiny:1.0.0",
                "--iterations",
                "5",
                "--baseline-file",
                str(baseline),
            ],
            monkeypatch,
        )
        code = _run(
            regression_mod,
            [
                "check",
                "--model",
                "tiny:1.0.0",
                "--iterations",
                "5",
                "--baseline-file",
                str(baseline),
                "--output",
                str(report),
            ],
            monkeypatch,
        )
        assert code in (0, 1)
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload and payload[0]["model"] == "tiny:1.0.0"

    def test_check_all_covers_every_recorded_model(
        self, registry: str, tmp_path: Path, monkeypatch
    ) -> None:
        baseline = tmp_path / "baseline.json"
        report = tmp_path / "regression.json"
        _run(
            regression_mod,
            [
                "record",
                "--model",
                "tiny:1.0.0",
                "--iterations",
                "5",
                "--baseline-file",
                str(baseline),
            ],
            monkeypatch,
        )
        code = _run(
            regression_mod,
            [
                "check-all",
                "--iterations",
                "5",
                "--baseline-file",
                str(baseline),
                "--output",
                str(report),
            ],
            monkeypatch,
        )
        assert code in (0, 1)
        assert json.loads(report.read_text(encoding="utf-8"))

    def test_a_model_that_cannot_be_measured_is_skipped_not_fatal(
        self, registry: str, tmp_path: Path, monkeypatch
    ) -> None:
        """One broken artifact must not abandon the rest of the sweep."""
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps({"baselines": {"gone:1.0.0": {"metrics": {"p95_latency_ms": 1.0}}}}),
            encoding="utf-8",
        )
        code = _run(
            regression_mod,
            [
                "check-all",
                "--iterations",
                "5",
                "--baseline-file",
                str(baseline),
                "--output",
                str(tmp_path / "r.json"),
            ],
            monkeypatch,
        )
        assert code == 0


class TestDetectDrift:
    def test_identical_samples_report_no_drift(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(0.8, 0.05, 500).tolist()
        report = detect_drift(
            model="tiny:1.0.0",
            reference_confidences=sample,
            current_confidences=sample,
        )
        assert str(report.overall_severity) == "none"
        assert report.drifted is False

    def test_a_shifted_distribution_is_caught(self) -> None:
        rng = np.random.default_rng(1)
        report = detect_drift(
            model="tiny:1.0.0",
            reference_confidences=rng.normal(0.9, 0.03, 500).tolist(),
            current_confidences=rng.normal(0.4, 0.03, 500).tolist(),
        )
        assert report.drifted is True

    def test_label_drift_is_detected(self) -> None:
        report = detect_drift(
            model="tiny:1.0.0",
            reference_labels=["cat"] * 400 + ["dog"] * 100,
            current_labels=["cat"] * 100 + ["dog"] * 400,
        )
        assert report.drifted is True

    def test_feature_drift_is_detected(self) -> None:
        rng = np.random.default_rng(2)
        report = detect_drift(
            model="tiny:1.0.0",
            reference_features={"brightness": rng.normal(120, 5, 400).tolist()},
            current_features={"brightness": rng.normal(200, 5, 400).tolist()},
        )
        assert report.drifted is True

    def test_the_worst_signal_wins_rather_than_the_average(self) -> None:
        """One badly drifted signal must not be diluted by several stable ones."""
        rng = np.random.default_rng(3)
        stable = rng.normal(0.8, 0.05, 500).tolist()
        report = detect_drift(
            model="tiny:1.0.0",
            reference_confidences=stable,
            current_confidences=stable,
            reference_features={"brightness": rng.normal(120, 5, 500).tolist()},
            current_features={"brightness": rng.normal(220, 5, 500).tolist()},
        )
        assert report.drifted is True

    def test_no_data_at_all_is_reported_not_crashed(self) -> None:
        report = detect_drift(model="tiny:1.0.0")
        assert report.results == []
        assert report.drifted is False

    def test_the_report_serialises(self) -> None:
        rng = np.random.default_rng(4)
        payload = detect_drift(
            model="tiny:1.0.0",
            reference_confidences=rng.normal(0.8, 0.05, 200).tolist(),
            current_confidences=rng.normal(0.8, 0.05, 200).tolist(),
        ).to_dict()
        json.dumps(payload)
        assert payload["model"] == "tiny:1.0.0"


class TestDriftCli:
    """The CLI reads from the inference log, so the database call is stubbed;
    everything downstream of it - printing, severity, the written report - is
    the real code path."""

    def test_writes_a_report_from_the_inference_log(self, tmp_path: Path, monkeypatch) -> None:
        rng = np.random.default_rng(7)
        report = detect_drift(
            model="tiny:1.0.0",
            reference_confidences=rng.normal(0.9, 0.03, 300).tolist(),
            current_confidences=rng.normal(0.4, 0.03, 300).tolist(),
        )

        async def fake_from_db(*a, **k):
            return report

        class _Db:
            async def connect(self):
                return True

        monkeypatch.setattr(drift_mod, "detect_drift_from_database", fake_from_db)
        monkeypatch.setattr(drift_mod, "get_db_service", lambda: _Db(), raising=False)
        monkeypatch.setitem(
            __import__("sys").modules,
            "api.services.db_service",
            type("M", (), {"get_db_service": staticmethod(lambda: _Db())}),
        )

        out = tmp_path / "drift.json"
        code = _run(
            drift_mod,
            [
                "--model",
                "tiny",
                "--reference-days",
                "30",
                "--current-days",
                "1",
                "--output",
                str(out),
            ],
            monkeypatch,
        )
        assert code == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["model"] == "tiny:1.0.0"
        assert written["results"], "the drift results were not persisted"


class TestCompare:
    @staticmethod
    def _scores(name: str, correct: list[bool], latency: float) -> ModelScores:
        return ModelScores(
            name=name,
            correct=correct,
            confidences=[0.9 if c else 0.4 for c in correct],
            latencies_ms=[latency] * len(correct),
        )

    def test_a_clearly_better_challenger_is_promoted(self) -> None:
        champion = self._scores("a", [True] * 60 + [False] * 140, 10.0)
        challenger = self._scores("b", [True] * 160 + [False] * 40, 10.0)
        result = compare(champion, challenger)
        assert result.significant is True
        assert "promote" in result.recommendation.lower()

    def test_a_tiny_win_is_not_worth_a_deployment(self) -> None:
        """Statistical significance is not the same as being worth shipping."""
        champion = self._scores("a", [True] * 100 + [False] * 100, 10.0)
        challenger = self._scores("b", [True] * 101 + [False] * 99, 10.0)
        result = compare(champion, challenger, min_improvement=0.10)
        assert "promote" not in result.recommendation.lower()

    def test_a_win_paid_for_in_latency_is_refused(self) -> None:
        champion = self._scores("a", [True] * 60 + [False] * 140, 10.0)
        challenger = self._scores("b", [True] * 160 + [False] * 40, 500.0)
        result = compare(champion, challenger, max_latency_regression_ms=50.0)
        assert "not promote" in result.recommendation.lower()
        assert "slower" in result.recommendation.lower()

    def test_unequal_sample_counts_are_refused(self) -> None:
        """An unpaired McNemar test is not a test."""
        with pytest.raises(ValueError, match="same samples"):
            compare(self._scores("a", [True] * 10, 1.0), self._scores("b", [True] * 9, 1.0))

    def test_the_result_serialises(self) -> None:
        result = compare(
            self._scores("a", [True] * 100 + [False] * 100, 10.0),
            self._scores("b", [True] * 120 + [False] * 80, 10.0),
        )
        json.dumps(result.to_dict())


class TestEvaluateModels:
    def test_both_models_see_identical_samples(self) -> None:
        samples = [(_png(i), i % 2) for i in range(10)]
        scores = evaluate_models(
            {
                "a": lambda b: (0, 0.9, 1.0),
                "b": lambda b: (1, 0.8, 2.0),
            },
            samples,
        )
        assert scores["a"].n == scores["b"].n == 10

    def test_a_predictor_that_raises_counts_as_wrong_not_as_missing(self) -> None:
        """Dropping the sample would invalidate the paired comparison."""
        samples = [(_png(i), 0) for i in range(6)]

        def explode(_: bytes):
            raise RuntimeError("model died")

        scores = evaluate_models({"ok": lambda b: (0, 0.9, 1.0), "bad": explode}, samples)
        assert scores["bad"].n == scores["ok"].n == 6
        assert scores["bad"].errors == 6
        assert scores["bad"].accuracy == 0.0


@pytest.fixture
def stub_dataset(tmp_path: Path, monkeypatch):
    """Stand in for Tiny-ImageNet so the A/B CLI can run without a 240 MB download.

    Only the three attributes `main()` touches are provided: the class list,
    the class_to_idx map, and `(path, label)` samples on disk.
    """
    import models.training.dataset as dataset_mod

    images = tmp_path / "images"
    images.mkdir()
    samples = []
    for i in range(8):
        path = images / f"{i}.png"
        path.write_bytes(_png(200 + i))
        samples.append((path, i % NUM_CLASSES))

    classes = [f"class_{i}" for i in range(NUM_CLASSES)]

    class _Train:
        def __init__(self, *a, **k):
            self.classes = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.samples = samples

    class _Val(_Train):
        pass

    monkeypatch.setattr(dataset_mod, "find_dataset_root", lambda *a, **k: images)
    monkeypatch.setattr(dataset_mod, "TinyImageNetTrain", _Train)
    monkeypatch.setattr(dataset_mod, "TinyImageNetVal", _Val)
    return images


class TestAbTestCli:
    def test_compares_two_registered_models_end_to_end(
        self, registry: str, stub_dataset, tmp_path: Path, monkeypatch
    ) -> None:
        out = tmp_path / "ab.json"
        code = _run(
            ab_mod,
            [
                "--champion",
                "tiny:1.0.0",
                "--challenger",
                "tiny-challenger:1.0.0",
                "--data-dir",
                str(stub_dataset),
                "--samples",
                "8",
                "--output",
                str(out),
            ],
            monkeypatch,
        )
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["champion"] == "tiny:1.0.0"
        assert payload["n_samples"] == 8

    def test_identical_models_are_not_promoted(
        self, registry: str, stub_dataset, tmp_path: Path, monkeypatch
    ) -> None:
        """Both entries point at the same artifact, so there is nothing to win."""
        out = tmp_path / "ab.json"
        _run(
            ab_mod,
            [
                "--champion",
                "tiny:1.0.0",
                "--challenger",
                "tiny-challenger:1.0.0",
                "--data-dir",
                str(stub_dataset),
                "--samples",
                "8",
                "--output",
                str(out),
            ],
            monkeypatch,
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["significant"] is False
        assert "keep tiny:1.0.0" in payload["recommendation"].lower()

    def test_a_mismatched_label_space_is_refused_before_inference(
        self, registry: str, stub_dataset, tmp_path: Path, monkeypatch
    ) -> None:
        """Refusing costs a second; discovering it after a 500-sample run does not."""
        monkeypatch.setattr(
            ab_mod,
            "assert_comparable_label_spaces",
            lambda *a, **k: (_ for _ in ()).throw(
                ab_mod.IncomparableLabelSpacesError("4 classes vs 200")
            ),
        )
        code = _run(
            ab_mod,
            [
                "--champion",
                "tiny:1.0.0",
                "--challenger",
                "tiny-challenger:1.0.0",
                "--data-dir",
                str(stub_dataset),
                "--samples",
                "8",
                "--output",
                str(tmp_path / "ab.json"),
            ],
            monkeypatch,
        )
        assert code == 1


class TestRequiredSampleSize:
    def test_a_smaller_effect_needs_more_samples(self) -> None:
        big = ab_mod._required_sample_size(0.70, 0.10)
        small = ab_mod._required_sample_size(0.70, 0.01)
        assert small > big > 0

    def test_no_effect_needs_no_samples(self) -> None:
        assert ab_mod._required_sample_size(0.70, 0.0) == 0


class TestSignificantButWorseChallenger:
    def test_a_significantly_worse_challenger_says_so(self) -> None:
        champion = ModelScores(
            name="a",
            correct=[True] * 160 + [False] * 40,
            confidences=[0.9] * 200,
            latencies_ms=[10.0] * 200,
        )
        challenger = ModelScores(
            name="b",
            correct=[True] * 60 + [False] * 140,
            confidences=[0.9] * 200,
            latencies_ms=[10.0] * 200,
        )
        result = compare(champion, challenger)
        assert "worse" in result.recommendation.lower()
        assert "preprocessing mismatch" in result.recommendation.lower()
