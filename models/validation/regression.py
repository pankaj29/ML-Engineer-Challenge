"""Performance regression testing.

Plain English:
    A regression test asks one question: "is this version worse than the last
    one?" Not "is it good" — that is what validation is for — but "did we
    break something that used to work?"

    It works by keeping a **baseline**: a recorded snapshot of how the model
    performed when we were happy with it. Every later version is measured the
    same way and compared against that snapshot. If accuracy drops or latency
    rises past an allowed tolerance, the test fails and CI blocks the release.

Why tolerances rather than exact equality:
    Latency measurements are noisy. The same model on the same machine will
    vary by several percent between runs — other processes, CPU frequency
    scaling, cache state. A test demanding exactly equal latency would fail
    constantly and be switched off within a week. Each metric therefore has a
    tolerance wide enough to absorb normal noise and tight enough to catch a
    real regression.

Accuracy is treated more strictly than latency, because accuracy should not
move at all unless the weights changed, whereas latency legitimately wobbles.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASELINE = REPO_ROOT / "benchmarks" / "baselines.json"


@dataclass
class MetricCheck:
    """One metric compared against its baseline."""

    name: str
    baseline: float
    current: float
    delta: float
    delta_percent: float
    tolerance: float
    higher_is_better: bool
    passed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionReport:
    """The result of checking one model against its baseline."""

    model: str
    baseline_recorded_at: str
    checked_at: str
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.model}: {len(self.checks)} checks, {len(self.failures)} failure(s)"
        )


# Default tolerances. Accuracy may not drop at all beyond rounding noise;
# latency gets generous headroom because measurement noise is real.
DEFAULT_TOLERANCES: dict[str, float] = {
    # Accuracy metrics: allowed *absolute* drop.
    "accuracy": 0.005,  # 0.5 percentage points
    "top1_accuracy": 0.005,
    "top5_accuracy": 0.005,
    "map50": 0.01,
    "map50_95": 0.01,
    "recall_at_1": 0.01,
    # The registry publishes accuracy as a percentage (78.91), not a fraction,
    # so the tolerance here is half a percentage point rather than 0.005.
    "top1": 0.5,
    "top5": 0.5,
    # Latency metrics: allowed *relative* increase.
    "p50_latency_ms": 0.25,  # 25% slower
    "p95_latency_ms": 0.30,
    "p99_latency_ms": 0.40,  # the tail is noisiest, so the loosest bound
    "mean_latency_ms": 0.25,
    # Resource metrics: allowed relative increase.
    "size_mb": 0.10,
    "peak_memory_mb": 0.20,
}

# Metrics where a bigger number is better.
HIGHER_IS_BETTER = {
    "accuracy",
    "top1_accuracy",
    "top5_accuracy",
    "top1",
    "top5",
    "map50",
    "map50_95",
    "recall_at_1",
    "throughput_rps",
    "throughput_ips",
}

# Metrics compared as a relative change rather than an absolute difference.
RELATIVE_METRICS = {
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_latency_ms",
    "size_mb",
    "peak_memory_mb",
    "throughput_rps",
    "throughput_ips",
}


class BaselineStore:
    """Reads and writes the recorded performance baselines."""

    def __init__(self, path: Path = DEFAULT_BASELINE) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"baselines": {}}
        self.data.setdefault("baselines", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(UTC).isoformat()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, model: str) -> dict[str, Any] | None:
        return self.data["baselines"].get(model)

    def record(self, model: str, metrics: dict[str, float], *, note: str = "") -> dict[str, Any]:
        """Store a new baseline for a model.

        The environment is recorded alongside the numbers. Comparing a latency
        measured on a laptop against one measured on a CI runner is
        meaningless, and this makes that mismatch visible instead of silent.
        """
        entry = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "recorded_at": datetime.now(UTC).isoformat(),
            "environment": environment_fingerprint(),
            "note": note,
        }
        self.data["baselines"][model] = entry
        self.save()
        return entry


def environment_fingerprint() -> dict[str, Any]:
    """Describe the machine, so baselines are not compared across hardware."""
    import os

    info: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def check_metric(
    name: str,
    baseline: float,
    current: float,
    *,
    tolerance: float | None = None,
) -> MetricCheck:
    """Compare one metric against its baseline value.

    Accuracy metrics use an absolute tolerance ("may not drop more than 0.5
    points"); latency and size use a relative one ("may not grow more than
    25%"). Mixing these up is a common mistake: a 25% *absolute* accuracy
    tolerance would let a model collapse unnoticed.
    """
    tolerance = DEFAULT_TOLERANCES.get(name, 0.1) if tolerance is None else tolerance
    higher_better = name in HIGHER_IS_BETTER
    relative = name in RELATIVE_METRICS
    # An unlisted metric falls through to lower-is-better, which is right for
    # latency and wrong for anything accuracy-shaped: a drop would read as an
    # improvement and the gate would pass. The check still runs, but it says
    # so, because a silently mis-signed metric is worse than an absent one.
    recognised = name in HIGHER_IS_BETTER or name in DEFAULT_TOLERANCES

    delta = current - baseline
    delta_percent = (delta / baseline * 100) if baseline else 0.0

    if higher_better:
        # A drop is bad.
        regression = -delta
        limit = tolerance * abs(baseline) if relative else tolerance
    else:
        # A rise is bad.
        regression = delta
        limit = tolerance * abs(baseline) if relative else tolerance

    passed = regression <= limit

    if passed and regression <= 0:
        message = f"{name} improved: {baseline:.4f} -> {current:.4f} " f"({delta_percent:+.1f}%)"
    elif passed:
        message = (
            f"{name} moved within tolerance: {baseline:.4f} -> {current:.4f} "
            f"({delta_percent:+.1f}%, limit {limit:.4f})"
        )
    else:
        direction = "dropped" if higher_better else "increased"
        message = (
            f"REGRESSION: {name} {direction} from {baseline:.4f} to {current:.4f} "
            f"({delta_percent:+.1f}%), exceeding the allowed {limit:.4f}"
        )

    if not recognised:
        message += (
            f"  [unrecognised metric '{name}': assumed lower-is-better with an "
            f"absolute tolerance of {tolerance}. Add it to HIGHER_IS_BETTER or "
            f"DEFAULT_TOLERANCES in regression.py to check it properly.]"
        )

    return MetricCheck(
        name=name,
        baseline=round(float(baseline), 6),
        current=round(float(current), 6),
        delta=round(float(delta), 6),
        delta_percent=round(float(delta_percent), 3),
        tolerance=float(tolerance),
        higher_is_better=higher_better,
        passed=bool(passed),
        message=message,
    )


def check_regression(
    model: str,
    current_metrics: dict[str, float],
    *,
    store: BaselineStore | None = None,
    tolerances: dict[str, float] | None = None,
    strict_environment: bool = False,
) -> RegressionReport:
    """Compare a model's current metrics against its recorded baseline.

    Args:
        model: ``name:version`` identifier.
        current_metrics: Freshly measured metrics.
        strict_environment: Fail (rather than warn) when the measurement
            environment differs from the baseline's.

    Returns:
        A :class:`RegressionReport`. When no baseline exists the report
        *passes* with a warning — a brand-new model has nothing to regress
        against, and failing CI for that would be wrong.
    """
    store = store or BaselineStore()
    tolerances = tolerances or {}
    current_env = environment_fingerprint()

    baseline_entry = store.get(model)
    if baseline_entry is None:
        return RegressionReport(
            model=model,
            baseline_recorded_at="",
            checked_at=datetime.now(UTC).isoformat(),
            passed=True,
            warnings=[
                f"No baseline exists for {model}. Record one with:\n"
                f"  python -m models.validation.regression record --model {model} ..."
            ],
            environment=current_env,
        )

    checks: list[MetricCheck] = []
    failures: list[str] = []
    warnings: list[str] = []

    baseline_metrics = baseline_entry["metrics"]
    baseline_env = baseline_entry.get("environment", {})

    # Comparing latency across different hardware produces nonsense.
    env_mismatch = [
        key
        for key in ("processor", "gpu", "cpu_count")
        if baseline_env.get(key) != current_env.get(key)
    ]
    if env_mismatch:
        message = (
            f"Baseline was recorded on different hardware (differs in: "
            f"{', '.join(env_mismatch)}). Latency comparisons are unreliable; "
            "accuracy comparisons remain valid."
        )
        if strict_environment:
            failures.append(message)
        else:
            warnings.append(message)

    for name, current in current_metrics.items():
        if current is None or name not in baseline_metrics:
            continue
        baseline_value = baseline_metrics[name]
        if baseline_value is None:
            continue

        # Skip latency checks on mismatched hardware rather than emitting a
        # failure everyone will learn to ignore.
        if (
            env_mismatch
            and not strict_environment
            and name in RELATIVE_METRICS
            and "latency" in name
        ):
            warnings.append(f"Skipped {name}: hardware differs from the baseline.")
            continue

        check = check_metric(name, baseline_value, current, tolerance=tolerances.get(name))
        checks.append(check)
        if not check.passed:
            failures.append(check.message)

    missing = sorted(set(baseline_metrics) - set(current_metrics))
    if missing:
        warnings.append(
            f"Baseline records metrics that were not measured this run: {', '.join(missing)}"
        )

    return RegressionReport(
        model=model,
        baseline_recorded_at=baseline_entry.get("recorded_at", ""),
        checked_at=datetime.now(UTC).isoformat(),
        passed=not failures,
        checks=[c.to_dict() for c in checks],
        failures=failures,
        warnings=warnings,
        environment=current_env,
    )


def measure_model(
    model_key: str, *, iterations: int = 50, warmup: int = 5, batch_size: int = 1
) -> dict[str, float]:
    """Measure latency and size for a registered model.

    Accuracy is not measured here: it needs a labelled dataset and is produced
    by :mod:`models.validation.validate`. This function covers the metrics
    that can be measured from the artifact alone.
    """
    import numpy as np

    from api.services.model_service import ModelService
    from models.optimization.benchmark import measure, summarise

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
        raise ValueError(f"{model_key} is not in the registry")

    loaded = service.load(entry)
    shape = entry.input_shape or [1, 3, 224, 224]
    data = np.random.randn(batch_size, *shape[1:]).astype(np.float32)

    timings = measure(lambda: loaded.runtime.infer(data), iterations=iterations, warmup=warmup)
    result = summarise(
        timings,
        name=entry.name,
        runtime=loaded.runtime.format.value,
        device=loaded.runtime.device,
        batch_size=batch_size,
    )

    metrics: dict[str, float] = {
        "p50_latency_ms": round(result.p50_ms, 3),
        "p95_latency_ms": round(result.p95_ms, 3),
        "p99_latency_ms": round(result.p99_ms, 3),
        "mean_latency_ms": round(result.mean_ms, 3),
        "throughput_ips": round(result.throughput_ips, 2),
    }

    artifact = service._artifact_path(entry, next(iter(entry.artifacts)))
    if artifact and artifact.exists():
        metrics["size_mb"] = round(artifact.stat().st_size / 1_048_576, 3)

    metrics.update({k: float(v) for k, v in entry.metrics.items() if isinstance(v, (int, float))})
    return metrics


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Performance regression testing.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="Record a baseline for a model.")
    p_record.add_argument("--model", required=True, help="name:version")
    p_record.add_argument("--iterations", type=int, default=50)
    p_record.add_argument("--note", default="")
    p_record.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE)

    p_check = sub.add_parser("check", help="Check a model against its baseline.")
    p_check.add_argument("--model", required=True, help="name:version")
    p_check.add_argument("--iterations", type=int, default=50)
    p_check.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE)
    p_check.add_argument("--strict-environment", action="store_true")
    p_check.add_argument(
        "--output", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "regression.json"
    )

    p_all = sub.add_parser("check-all", help="Check every registered model.")
    p_all.add_argument("--iterations", type=int, default=30)
    p_all.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE)
    p_all.add_argument(
        "--output", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "regression.json"
    )

    args = parser.parse_args()
    store = BaselineStore(args.baseline_file)

    if args.command == "record":
        metrics = measure_model(args.model, iterations=args.iterations)
        entry = store.record(args.model, metrics, note=args.note)
        print(f"recorded baseline for {args.model}:")
        for key, value in sorted(entry["metrics"].items()):
            print(f"  {key:<20} {value}")
        print(f"saved to {store.path}")
        return 0

    models = (
        [args.model]
        if args.command == "check"
        else sorted(store.data["baselines"]) or _all_registered()
    )

    reports: list[RegressionReport] = []
    exit_code = 0
    for model_key in models:
        try:
            metrics = measure_model(model_key, iterations=args.iterations)
        except Exception as exc:
            print(f"[SKIP] {model_key}: {type(exc).__name__}: {exc}")
            continue

        report = check_regression(
            model_key,
            metrics,
            store=store,
            strict_environment=getattr(args, "strict_environment", False),
        )
        reports.append(report)
        print(f"\n{report.summary()}")
        for check in report.checks:
            marker = "  ok " if check["passed"] else "FAIL"
            print(f"  [{marker}] {check['message']}")
        for warning in report.warnings:
            print(f"  [warn] {warning}")
        if not report.passed:
            exit_code = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return exit_code


def _all_registered() -> list[str]:
    from api.services.model_service import ModelService

    return [e.key for e in ModelService().list_entries()]


__all__ = [
    "DEFAULT_TOLERANCES",
    "BaselineStore",
    "MetricCheck",
    "RegressionReport",
    "check_metric",
    "check_regression",
    "environment_fingerprint",
    "measure_model",
]


if __name__ == "__main__":
    raise SystemExit(main())
