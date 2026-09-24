"""Comprehensive model validation pipeline.

Plain English:
    Before a model is allowed to serve real traffic, it has to pass a series
    of checks. This module is that gate. It is deliberately more paranoid than
    "what is the accuracy", because most production model failures are not
    accuracy failures — they are plumbing failures that a single accuracy
    number would never catch.

The checks, in the order they run (cheapest and most fundamental first):

1. **Artifact integrity** — do the files exist, load, and have the shape the
   registry claims?
2. **Determinism** — does the same input give the same output twice? A model
   that does not is either using dropout in eval mode or has a non-deterministic
   kernel, and any accuracy figure from it is unreliable.
3. **Batch invariance** — does an image predicted alone give the same answer
   inside a batch of eight? This catches the classic bug of a model in
   ``train()`` mode, where BatchNorm uses batch statistics and every
   prediction depends on its neighbours.
4. **Output sanity** — do the probabilities sum to 1? Are there NaNs?
5. **Robustness** — does a tiny, invisible change to the input flip the
   prediction? A model that is that brittle will behave erratically on real
   photos from different cameras.
6. **Accuracy** — top-1 and top-5 on a held-out set.
7. **Calibration** — when the model says 90%, is it right 90% of the time?
   Measured with **Expected Calibration Error**. Overconfident models are
   dangerous precisely because downstream code trusts the confidence.
8. **Latency** — does it meet the sub-second requirement?

Any check can fail without failing the whole run: the report lists every
result so a reviewer sees the complete picture rather than only the first
problem.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CheckResult:
    """Outcome of one validation check."""

    name: str
    passed: bool
    severity: str  # "critical" blocks release; "warning" does not
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """The full validation outcome for one model."""

    model: str
    validated_at: str
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.model}: {len(self.checks)} checks, "
            f"{len(self.critical_failures)} critical failure(s), "
            f"{len(self.warnings)} warning(s)"
        )


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_artifacts(entry: Any, artifacts_dir: Path) -> CheckResult:
    """Confirm every registered artifact file exists and is non-empty."""
    missing: list[str] = []
    sizes: dict[str, float] = {}

    for fmt, rel in entry.artifacts.items():
        path = Path(rel)
        if not path.is_absolute():
            path = artifacts_dir / path
        if not path.exists():
            missing.append(f"{fmt}: {path}")
        elif path.stat().st_size == 0:
            missing.append(f"{fmt}: {path} is empty")
        else:
            sizes[fmt] = round(path.stat().st_size / 1_048_576, 2)

    if missing:
        return CheckResult(
            name="artifact_integrity",
            passed=False,
            severity="critical",
            message=f"{len(missing)} artifact(s) missing or empty.",
            detail={"missing": missing, "present": sizes},
        )
    return CheckResult(
        name="artifact_integrity",
        passed=True,
        severity="critical",
        message=f"All {len(sizes)} artifact(s) present.",
        detail={"sizes_mb": sizes},
    )


def check_determinism(
    infer: Callable[[np.ndarray], list[np.ndarray]],
    sample: np.ndarray,
    *,
    runs: int = 3,
    tolerance: float = 1e-5,
) -> CheckResult:
    """Run the same input several times and require identical output.

    A non-deterministic model makes every other measurement meaningless: you
    can no longer tell whether an accuracy change came from a code change or
    from randomness.
    """
    outputs = [infer(sample)[0] for _ in range(runs)]
    diffs = [float(np.abs(outputs[0] - o).max()) for o in outputs[1:]]
    max_diff = max(diffs) if diffs else 0.0

    passed = max_diff <= tolerance
    return CheckResult(
        name="determinism",
        passed=passed,
        severity="critical",
        message=(
            f"Output is deterministic across {runs} runs (max diff {max_diff:.2e})."
            if passed
            else (
                f"Output is NOT deterministic: max difference {max_diff:.2e} across "
                f"{runs} identical runs. Check the model is in eval mode and no "
                "dropout or non-deterministic kernel is active."
            )
        ),
        detail={"runs": runs, "max_diff": max_diff, "tolerance": tolerance},
    )


def check_batch_invariance(
    infer: Callable[[np.ndarray], list[np.ndarray]],
    sample: np.ndarray,
    *,
    batch_size: int = 8,
    tolerance: float = 1e-3,
) -> CheckResult:
    """Require that a prediction does not depend on batch position.

    This is the check that catches a model accidentally left in ``train()``
    mode. In training mode BatchNorm normalises using the statistics of the
    current batch, so an image's prediction changes depending on what else was
    in the batch — which shows up in production as intermittently wrong
    answers that nobody can reproduce.
    """
    single = infer(sample)[0]

    batched_input = np.repeat(sample, batch_size, axis=0)
    try:
        batched = infer(batched_input)[0]
    except Exception as exc:
        return CheckResult(
            name="batch_invariance",
            passed=False,
            severity="critical",
            message=(
                f"The model failed on a batch of {batch_size}: {type(exc).__name__}. "
                "The exported graph likely has a fixed batch dimension."
            ),
            detail={"error": str(exc)[:300]},
        )

    diffs = [float(np.abs(single[0] - batched[i]).max()) for i in range(batch_size)]
    max_diff = max(diffs)
    passed = max_diff <= tolerance

    return CheckResult(
        name="batch_invariance",
        passed=passed,
        severity="critical",
        message=(
            f"Predictions are batch-invariant (max diff {max_diff:.2e})."
            if passed
            else (
                f"Predictions CHANGE with batch position (max diff {max_diff:.2e}). "
                "The most likely cause is the model being in train() mode, so "
                "BatchNorm uses batch statistics."
            )
        ),
        detail={"batch_size": batch_size, "max_diff": max_diff},
    )


def check_output_sanity(
    infer: Callable[[np.ndarray], list[np.ndarray]],
    sample: np.ndarray,
    *,
    expect_probabilities: bool = True,
) -> CheckResult:
    """Check the output for NaNs, infinities and malformed probabilities."""
    output = infer(sample)[0]
    problems: list[str] = []

    if np.isnan(output).any():
        problems.append(f"{int(np.isnan(output).sum())} NaN values in the output")
    if np.isinf(output).any():
        problems.append(f"{int(np.isinf(output).sum())} infinite values in the output")

    detail: dict[str, Any] = {
        "shape": list(output.shape),
        "min": float(np.nanmin(output)) if output.size else 0.0,
        "max": float(np.nanmax(output)) if output.size else 0.0,
    }

    if expect_probabilities and output.ndim == 2 and not problems:
        # Check the RAW output, not softmax(output).
        #
        # This previously computed softmax(output) and then asserted the result
        # summed to 1. Softmax always sums to 1 by construction, so the branch
        # could never fail - a check that cannot detect its own failure mode.
        #
        # A classifier here may legitimately emit either probabilities or raw
        # logits, so the rule is: if it looks like probabilities, it must be
        # valid probabilities. Anything outside [0, 1] is treated as logits and
        # only sanity-checked for finiteness (already done above).
        raw_total = float(output.astype(np.float64).sum(axis=1)[0])
        looks_like_probabilities = bool(output.min() >= -1e-6 and output.max() <= 1.0 + 1e-6)
        detail["raw_sum"] = round(raw_total, 6)
        detail["looks_like_probabilities"] = looks_like_probabilities

        if looks_like_probabilities and abs(raw_total - 1.0) > 1e-3:
            problems.append(
                f"output is in [0, 1] so it should be a probability "
                f"distribution, but it sums to {raw_total:.6f}, not 1.0"
            )

        # Report the normalised view either way - it is what the API returns.
        probs = softmax(output.astype(np.float64))
        detail["max_probability"] = round(float(probs.max()), 6)

    passed = not problems
    return CheckResult(
        name="output_sanity",
        passed=passed,
        severity="critical",
        message=(
            "Output is well-formed: no NaN or infinite values."
            if passed
            else "Output is malformed: " + "; ".join(problems)
        ),
        detail=detail,
    )


def check_robustness(
    infer: Callable[[np.ndarray], list[np.ndarray]],
    samples: list[np.ndarray],
    *,
    noise_std: float = 0.01,
    max_flip_rate: float = 0.15,
) -> CheckResult:
    """Add imperceptible noise and count how often the prediction changes.

    ``noise_std`` of 0.01 in normalised space is far below what a human can
    see — roughly the difference between two photos of the same scene. If a
    meaningful fraction of predictions flip, the model is sitting right on its
    decision boundary and will behave inconsistently on real inputs from
    different cameras or compression settings.

    This is a **warning**, not a critical failure: some brittleness is normal,
    especially for fine-grained classes.
    """
    if not samples:
        return CheckResult(
            name="robustness",
            passed=True,
            severity="warning",
            message="Skipped: no samples supplied.",
        )

    flips = 0
    confidence_drops: list[float] = []

    for sample in samples:
        clean = infer(sample)[0]
        noisy_input = sample + np.random.normal(0, noise_std, sample.shape).astype(np.float32)
        noisy = infer(noisy_input)[0]

        if clean.ndim == 2 and clean.shape[1] > 1:
            clean_class = int(clean[0].argmax())
            noisy_class = int(noisy[0].argmax())
            if clean_class != noisy_class:
                flips += 1
            clean_conf = float(softmax(clean.astype(np.float64))[0].max())
            noisy_conf = float(softmax(noisy.astype(np.float64))[0].max())
            confidence_drops.append(clean_conf - noisy_conf)

    flip_rate = flips / len(samples)
    passed = flip_rate <= max_flip_rate

    return CheckResult(
        name="robustness",
        passed=passed,
        severity="warning",
        message=(
            f"{flip_rate:.1%} of predictions changed under imperceptible noise "
            f"(sigma={noise_std}), within the {max_flip_rate:.0%} threshold."
            if passed
            else (
                f"BRITTLE: {flip_rate:.1%} of predictions flipped under imperceptible "
                f"noise (sigma={noise_std}), above the {max_flip_rate:.0%} threshold. "
                "Expect inconsistent results on visually similar real-world inputs."
            )
        ),
        detail={
            "samples": len(samples),
            "flips": flips,
            "flip_rate": round(flip_rate, 4),
            "mean_confidence_drop": (
                round(float(np.mean(confidence_drops)), 4) if confidence_drops else 0.0
            ),
        },
    )


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> tuple[float, list[dict[str, Any]]]:
    """Expected Calibration Error, with the per-bin breakdown.

    Sort predictions into confidence buckets. In each bucket, compare the
    average confidence against the actual accuracy. ECE is the average gap,
    weighted by how many predictions fall in each bucket.

    A perfectly calibrated model has ECE of 0: when it says 70%, it is right
    70% of the time. Modern deep networks are typically *overconfident*, with
    ECE between 0.05 and 0.15, which matters as soon as anything downstream
    thresholds on the confidence value.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=bool)

    if conf.size == 0:
        return 0.0, []

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    breakdown: list[dict[str, Any]] = []

    for i in range(bins):
        lower, upper = edges[i], edges[i + 1]
        # Upper-inclusive on the final bin so confidence exactly 1.0 is counted.
        mask = (conf > lower) & (conf <= upper) if i > 0 else (conf >= lower) & (conf <= upper)
        count = int(mask.sum())
        if count == 0:
            continue

        bin_conf = float(conf[mask].mean())
        bin_acc = float(corr[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (count / conf.size) * gap

        breakdown.append(
            {
                "range": f"({lower:.1f}, {upper:.1f}]",
                "count": count,
                "mean_confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "gap": round(gap, 4),
                "direction": "overconfident" if bin_conf > bin_acc else "underconfident",
            }
        )

    return float(ece), breakdown


def check_calibration(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    max_ece: float = 0.15,
) -> CheckResult:
    """Check how well confidence matches observed accuracy."""
    if len(confidences) < 50:
        return CheckResult(
            name="calibration",
            passed=True,
            severity="warning",
            message=f"Skipped: only {len(confidences)} samples (50 needed).",
        )

    ece, breakdown = expected_calibration_error(confidences, correct)
    passed = ece <= max_ece

    overconfident = sum(1 for b in breakdown if b["direction"] == "overconfident")
    return CheckResult(
        name="calibration",
        passed=passed,
        severity="warning",
        message=(
            f"Expected Calibration Error is {ece:.4f}, within the {max_ece} threshold."
            if passed
            else (
                f"POORLY CALIBRATED: ECE is {ece:.4f}, above the {max_ece} threshold. "
                f"{overconfident} of {len(breakdown)} confidence bands are overconfident. "
                "Do not threshold on raw confidence without recalibrating first."
            )
        ),
        detail={"ece": round(ece, 6), "bins": breakdown},
    )


def check_latency(
    infer: Callable[[np.ndarray], list[np.ndarray]],
    sample: np.ndarray,
    *,
    iterations: int = 30,
    warmup: int = 5,
    max_p95_ms: float = 1000.0,
) -> CheckResult:
    """Measure latency and check it against the sub-second requirement."""
    for _ in range(warmup):
        infer(sample)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer(sample)
        timings.append((time.perf_counter() - start) * 1000)

    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
    passed = p95 <= max_p95_ms

    return CheckResult(
        name="latency",
        passed=passed,
        severity="critical",
        message=(
            f"p95 latency is {p95:.1f} ms, within the {max_p95_ms:.0f} ms budget."
            if passed
            else f"TOO SLOW: p95 latency is {p95:.1f} ms, above the {max_p95_ms:.0f} ms budget."
        ),
        detail={
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(timings[0], 2),
            "max_ms": round(timings[-1], 2),
            "iterations": iterations,
        },
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def validate_model(
    model_key: str,
    *,
    eval_samples: list[tuple[bytes, int]] | None = None,
    max_p95_ms: float = 1000.0,
    robustness_samples: int = 20,
) -> ValidationReport:
    """Run the full validation suite against a registered model.

    Args:
        model_key: ``name:version``.
        eval_samples: Labelled ``(image_bytes, label)`` pairs. Accuracy and
            calibration checks are skipped when omitted.
        max_p95_ms: Latency budget.

    Returns:
        A :class:`ValidationReport`. ``passed`` is False only when a
        *critical* check failed; warnings do not block a release.
    """
    from api.services.model_service import ModelService
    from api.utils.image_processing import preprocess

    service = ModelService()
    name, _, version = model_key.partition(":")

    entry = next(
        (
            e
            for e in service.list_entries()
            if e.name == name and (not version or e.version == version)
        ),
        None,
    )
    if entry is None:
        return ValidationReport(
            model=model_key,
            validated_at=datetime.now(UTC).isoformat(),
            passed=False,
            critical_failures=[f"{model_key} is not in the registry"],
        )

    checks: list[CheckResult] = []
    metrics: dict[str, float] = {}

    # --- 1. Artifacts -----------------------------------------------------
    artifact_check = check_artifacts(entry, Path(service.artifacts_dir))
    checks.append(artifact_check)
    if not artifact_check.passed:
        return _finalise(model_key, checks, metrics)

    loaded = service.load(entry)
    infer = loaded.runtime.infer
    cfg = loaded.preprocess_config
    shape = entry.input_shape or [1, 3, cfg.size[0], cfg.size[1]]
    sample = np.random.randn(1, *shape[1:]).astype(np.float32)

    is_classifier = entry.task.value in ("classification",)

    # --- 2-4. Behavioural checks -----------------------------------------
    checks.append(check_determinism(infer, sample))
    checks.append(check_batch_invariance(infer, sample))
    checks.append(check_output_sanity(infer, sample, expect_probabilities=is_classifier))

    # --- 5. Robustness ----------------------------------------------------
    if is_classifier:
        noise_samples = [
            np.random.randn(1, *shape[1:]).astype(np.float32) for _ in range(robustness_samples)
        ]
        checks.append(check_robustness(infer, noise_samples))

    # --- 6-7. Accuracy and calibration ------------------------------------
    # Accuracy is only meaningful when the model's label space matches the
    # evaluation set's. Comparing an ImageNet-1k model (1000 classes) against
    # Tiny-ImageNet labels (200 classes) compares unrelated integers and
    # produces a confident 0% that looks like a broken model but is really a
    # broken measurement. Refuse to report that number.
    label_space_ok = True
    if eval_samples and is_classifier:
        dataset_classes = max(label for _, label in eval_samples) + 1
        model_classes = entry.num_classes or 0
        if model_classes and dataset_classes > model_classes:
            label_space_ok = False
        elif model_classes and model_classes > dataset_classes * 2:
            # The model predicts far more classes than the dataset contains,
            # so its indices cannot line up with the dataset's.
            label_space_ok = False

        if not label_space_ok:
            checks.append(
                CheckResult(
                    name="accuracy",
                    passed=True,
                    severity="warning",
                    message=(
                        f"Accuracy NOT measured: the model has {model_classes} classes but the "
                        f"evaluation set uses {dataset_classes}. Their class indices refer to "
                        "different things, so any accuracy figure would be meaningless. "
                        "Evaluate against a dataset matching the model's label space, or "
                        "fine-tune the model on this one."
                    ),
                    detail={
                        "model_classes": model_classes,
                        "dataset_classes": dataset_classes,
                    },
                )
            )

    if eval_samples and is_classifier and label_space_ok:
        correct: list[bool] = []
        top5_correct: list[bool] = []
        confidences: list[float] = []

        for image_bytes, label in eval_samples:
            try:
                arr = preprocess(image_bytes, cfg).array
                logits = infer(arr)[0]
                probs = softmax(logits.astype(np.float64))[0]
                ranked = np.argsort(probs)[::-1]
                correct.append(int(ranked[0]) == label)
                top5_correct.append(label in ranked[:5].tolist())
                confidences.append(float(probs.max()))
            except Exception:
                correct.append(False)
                top5_correct.append(False)
                confidences.append(0.0)

        top1 = float(np.mean(correct))
        top5 = float(np.mean(top5_correct))
        metrics["accuracy"] = round(top1, 6)
        metrics["top1_accuracy"] = round(top1, 6)
        metrics["top5_accuracy"] = round(top5, 6)
        metrics["mean_confidence"] = round(float(np.mean(confidences)), 6)

        checks.append(
            CheckResult(
                name="accuracy",
                passed=True,  # informational; thresholds belong to regression testing
                severity="warning",
                message=f"Top-1 {top1:.2%}, top-5 {top5:.2%} on {len(eval_samples)} samples.",
                detail={"samples": len(eval_samples), "top1": top1, "top5": top5},
            )
        )
        checks.append(check_calibration(confidences, correct))
        ece, _ = expected_calibration_error(confidences, correct)
        metrics["expected_calibration_error"] = round(ece, 6)

    # --- 8. Latency -------------------------------------------------------
    latency_check = check_latency(infer, sample, max_p95_ms=max_p95_ms)
    checks.append(latency_check)
    metrics["p50_latency_ms"] = latency_check.detail["p50_ms"]
    metrics["p95_latency_ms"] = latency_check.detail["p95_ms"]

    return _finalise(model_key, checks, metrics)


def _finalise(
    model_key: str, checks: list[CheckResult], metrics: dict[str, float]
) -> ValidationReport:
    """Assemble the report and decide the overall verdict."""
    critical = [c.message for c in checks if not c.passed and c.severity == "critical"]
    warnings = [c.message for c in checks if not c.passed and c.severity == "warning"]

    return ValidationReport(
        model=model_key,
        validated_at=datetime.now(UTC).isoformat(),
        passed=not critical,
        checks=[c.to_dict() for c in checks],
        metrics=metrics,
        critical_failures=critical,
        warnings=warnings,
    )


def load_eval_samples(data_dir: Path, limit: int = 200) -> list[tuple[bytes, int]]:
    """Load labelled validation images from Tiny-ImageNet."""
    from models.training.dataset import TinyImageNetTrain, TinyImageNetVal, find_dataset_root

    root = find_dataset_root(data_dir)
    train = TinyImageNetTrain(root)
    val = TinyImageNetVal(root, train.class_to_idx)
    return [(path.read_bytes(), label) for path, label in val.samples[:limit]]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a registered model.")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model to validate as name:version. Repeatable. Defaults to all.",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="Number of labelled images for accuracy/calibration. 0 skips those checks.",
    )
    parser.add_argument("--max-p95-ms", type=float, default=1000.0)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "validation.json"
    )
    args = parser.parse_args()

    from api.services.model_service import ModelService

    targets = args.model or [e.key for e in ModelService().list_entries()]

    eval_samples = None
    if args.eval_samples:
        try:
            eval_samples = load_eval_samples(args.data_dir, args.eval_samples)
            print(f"loaded {len(eval_samples)} labelled evaluation images\n")
        except Exception as exc:
            print(f"warning: could not load evaluation data ({exc}); skipping accuracy checks\n")

    reports: list[ValidationReport] = []
    exit_code = 0

    for key in targets:
        report = validate_model(key, eval_samples=eval_samples, max_p95_ms=args.max_p95_ms)
        reports.append(report)
        print(report.summary())
        for check in report.checks:
            marker = (
                " ok "
                if check["passed"]
                else ("FAIL" if check["severity"] == "critical" else "warn")
            )
            print(f"  [{marker}] {check['name']:<20} {check['message']}")
        print()
        if not report.passed:
            exit_code = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
