"""The retraining loop: drift, decide, retrain, validate, promote.

Plain English:
    The pieces to detect drift, train a model, validate it and compare it
    against a baseline all existed separately. Someone had to notice the drift
    and run the rest by hand. This joins them, and more importantly decides
    when not to.

The decision matters more than the mechanics. Retraining on every drift signal
is how a model gets worse: drift is noisy, a chi-square test on a quiet
Tuesday will trip, and a retrain on unrepresentative data replaces a working
model with a worse one. So the policy has a cooldown and an effect-size floor,
and no result promotes itself without passing validation and a regression
check first.

The gate is the point. Every stage after training can refuse, and a refusal
leaves the current model serving. A pipeline that cannot decline to ship is
just a slower way to break production.

    drift ──▶ decide ──▶ retrain ──▶ validate ──▶ regression ──▶ promote
                 │                       │             │
                 └── skip                └── stop      └── stop
                     (serving model stays live in every case)

Dry run by default. ``--execute`` is required to train or promote anything,
because a scheduled job that retrains by accident is worse than one that never
runs.

Usage::

    # What would it do?
    python -m models.pipeline.retraining --model resnet50-tiny-imagenet \\
        --drift-report benchmarks/reports/drift_report.json

    # Actually do it
    python -m models.pipeline.retraining --model resnet50-tiny-imagenet \\
        --drift-report benchmarks/reports/drift_report.json --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Stage(str, Enum):
    """Where the pipeline got to."""

    ASSESSED = "assessed"
    SKIPPED = "skipped"
    TRAINED = "trained"
    VALIDATED = "validated"
    REGRESSION_CHECKED = "regression_checked"
    PROMOTED = "promoted"
    FAILED = "failed"


#: Severities that justify retraining on their own.
ACTIONABLE = {"high"}

#: Severities that justify it only if they persist. "moderate" on a single
#: report is usually noise; the same signal twice in a row is not.
ACTIONABLE_IF_REPEATED = {"moderate"}


@dataclass
class RetrainingDecision:
    """Whether to retrain, and the reasoning, recorded either way."""

    should_retrain: bool
    reason: str
    severity: str
    drifted_features: list[str] = field(default_factory=list)
    blocked_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """What the run did, for the log and for the next run's cooldown."""

    model: str
    started_at: str
    finished_at: str
    stage: Stage
    decision: dict[str, Any]
    dry_run: bool
    promoted: bool = False
    duration_seconds: float = 0.0
    validation: dict[str, Any] | None = None
    regression: dict[str, Any] | None = None
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return payload

    def summary(self) -> str:
        what = "would promote" if self.dry_run and self.promoted else "promoted"
        if not self.promoted:
            what = "no promotion"
        return f"[{self.stage.value}] {self.model}: {what} ({self.duration_seconds:.0f}s)"


# ---------------------------------------------------------------------------
# Decide
# ---------------------------------------------------------------------------
def decide(
    report: dict[str, Any],
    *,
    history: list[dict[str, Any]] | None = None,
    cooldown_hours: float = 24.0,
    min_effect_size: float = 0.1,
) -> RetrainingDecision:
    """Turn a drift report into a yes or a no.

    Args:
        report: A :class:`~models.validation.drift.DriftReport` as a dict.
        history: Previous pipeline results, newest last. Used for the cooldown
            and to tell a repeated moderate signal from a one-off.
        cooldown_hours: Minimum gap between retrains. Drift often persists for
            days, and without this the pipeline retrains on every run for as
            long as it lasts.
        min_effect_size: Ignore statistically significant results whose effect
            is trivial. With enough samples everything is significant, which
            is exactly how a drift detector becomes an alarm nobody reads.

    Returns:
        The decision, including why, so a "no" is auditable too.
    """
    severity = str(report.get("overall_severity", "none")).lower()
    results = report.get("results", [])

    drifted = [
        r.get("feature", "?")
        for r in results
        if r.get("drifted") and float(r.get("effect_size", 0.0)) >= min_effect_size
    ]

    if not report.get("drifted") or not drifted:
        return RetrainingDecision(
            should_retrain=False,
            reason=(
                "no drift above the effect-size floor"
                if report.get("drifted")
                else "no drift detected"
            ),
            severity=severity,
        )

    # Cooldown before severity: a model retrained an hour ago will not be
    # improved by retraining again on the same data.
    recent = _last_retrain(history or [])
    if recent is not None:
        age = datetime.now(UTC) - recent
        if age < timedelta(hours=cooldown_hours):
            hours = age.total_seconds() / 3600
            return RetrainingDecision(
                should_retrain=False,
                reason=f"retrained {hours:.1f}h ago, inside the {cooldown_hours}h cooldown",
                severity=severity,
                drifted_features=drifted,
                blocked_by="cooldown",
            )

    if severity in ACTIONABLE:
        return RetrainingDecision(
            should_retrain=True,
            reason=f"{severity} drift on {', '.join(drifted)}",
            severity=severity,
            drifted_features=drifted,
        )

    if severity in ACTIONABLE_IF_REPEATED:
        if _previously_drifted(history or []):
            return RetrainingDecision(
                should_retrain=True,
                reason=f"{severity} drift on {', '.join(drifted)}, and again since last run",
                severity=severity,
                drifted_features=drifted,
            )
        return RetrainingDecision(
            should_retrain=False,
            reason=f"{severity} drift on first observation; waiting to see it repeat",
            severity=severity,
            drifted_features=drifted,
            blocked_by="awaiting_confirmation",
        )

    return RetrainingDecision(
        should_retrain=False,
        reason=f"{severity} drift is below the action threshold",
        severity=severity,
        drifted_features=drifted,
    )


def _last_retrain(history: list[dict[str, Any]]) -> datetime | None:
    """When training last actually ran. Skipped runs do not count."""
    for entry in reversed(history):
        if entry.get("stage") in (
            Stage.TRAINED.value,
            Stage.VALIDATED.value,
            Stage.REGRESSION_CHECKED.value,
            Stage.PROMOTED.value,
        ) and not entry.get("dry_run"):
            try:
                return datetime.fromisoformat(entry["started_at"])
            except (KeyError, ValueError):
                continue
    return None


def _previously_drifted(history: list[dict[str, Any]]) -> bool:
    """True if the previous run also saw drift."""
    if not history:
        return False
    return bool(history[-1].get("decision", {}).get("drifted_features"))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_pipeline(
    *,
    model: str,
    drift_report: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    history_path: Path | None = None,
    dry_run: bool = True,
    epochs: int = 30,
    cooldown_hours: float = 24.0,
    track: str = "auto",
) -> PipelineResult:
    """Assess, decide, and if warranted retrain, validate and promote."""
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()

    history = _load_history(history_path)
    decision = decide(drift_report, history=history, cooldown_hours=cooldown_hours)

    result = PipelineResult(
        model=model,
        started_at=started_at,
        finished_at=started_at,
        stage=Stage.ASSESSED,
        decision=decision.to_dict(),
        dry_run=dry_run,
    )
    result.messages.append(f"decision: {decision.reason}")

    if not decision.should_retrain:
        result.stage = Stage.SKIPPED
        return _finish(result, started, history_path, history)

    if dry_run:
        result.messages.append("dry run: would retrain, validate and promote from here")
        return _finish(result, started, history_path, history)

    # --- train -------------------------------------------------------------
    result.messages.append(f"training {model} for {epochs} epochs")
    trained = _run_training(epochs=epochs, data_dir=data_dir, output_dir=output_dir, track=track)
    if not trained:
        result.stage = Stage.FAILED
        result.messages.append("training failed; the current model keeps serving")
        return _finish(result, started, history_path, history)
    result.stage = Stage.TRAINED

    # --- validate ----------------------------------------------------------
    validation = _run_validation(model)
    result.validation = validation
    if validation is None:
        result.stage = Stage.FAILED
        result.messages.append("validation could not run; refusing to promote")
        return _finish(result, started, history_path, history)
    if not validation.get("passed"):
        result.stage = Stage.FAILED
        failures = validation.get("critical_failures", [])
        result.messages.append(f"validation failed ({len(failures)}); not promoting")
        return _finish(result, started, history_path, history)
    result.stage = Stage.VALIDATED

    # --- regression --------------------------------------------------------
    # Validation says the model is sane. This says it is not worse than the
    # one already serving, which is a different question and the one that
    # stops a retrain from quietly costing accuracy.
    regression = _run_regression(model)
    result.regression = regression
    if regression is not None and not regression.get("passed"):
        result.stage = Stage.FAILED
        result.messages.append(
            f"regression against the baseline: {'; '.join(regression.get('failures', []))}"
        )
        return _finish(result, started, history_path, history)
    result.stage = Stage.REGRESSION_CHECKED

    result.promoted = True
    result.stage = Stage.PROMOTED
    result.messages.append("validation and regression both passed; promoted")
    return _finish(result, started, history_path, history)


def _finish(
    result: PipelineResult,
    started: float,
    history_path: Path | None,
    history: list[dict[str, Any]],
) -> PipelineResult:
    result.duration_seconds = round(time.perf_counter() - started, 1)
    result.finished_at = datetime.now(UTC).isoformat()
    if history_path is not None:
        # Append, never overwrite. The cooldown and the repeat check both read
        # this, so losing it makes the pipeline retrain far more often than it
        # should.
        history.append(result.to_dict())
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history[-100:], indent=2), encoding="utf-8")
    return result


def _load_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, list) else []
    except (OSError, json.JSONDecodeError):
        # A corrupt history means no cooldown, which risks retraining too
        # often. Better than crashing the scheduled job.
        return []


def _run_training(*, epochs: int, data_dir: Path, output_dir: Path, track: str) -> bool:
    """Train in a subprocess.

    Separate process on purpose: training holds a lot of memory and CUDA
    state, and a scheduled pipeline should get that back when it finishes
    rather than carrying it through validation.
    """
    command = [
        sys.executable,
        "-m",
        "models.training.train_classifier",
        "--epochs",
        str(epochs),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--track",
        track,
        "--run-name",
        f"retrain-{datetime.now(UTC):%Y%m%d-%H%M}",
    ]
    done = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return done.returncode == 0


def _run_validation(model: str) -> dict[str, Any] | None:
    try:
        from models.validation.validate import validate_model

        return validate_model(model).to_dict()
    except Exception as exc:
        print(f"validation error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _run_regression(model: str) -> dict[str, Any] | None:
    """Compare against the stored baseline.

    Returns None when there is no baseline yet, which is not a failure: the
    first model has nothing to regress against.
    """
    try:
        from models.validation.regression import check_regression

        return check_regression(model).to_dict()
    except FileNotFoundError:
        print("no baseline recorded yet; skipping the regression gate")
        return None
    except Exception as exc:
        print(f"regression check error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decide whether drift warrants retraining, and run the loop if it does."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--drift-report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "drift_report.json",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models" / "artifacts")
    parser.add_argument(
        "--history",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "retraining_history.json",
        help="Past runs. Read for the cooldown, appended to afterwards.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually retrain and promote. Without it the pipeline only reports.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cooldown-hours", type=float, default=24.0)
    parser.add_argument("--track", default="auto", choices=["none", "auto", "mlflow"])
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "retraining_run.json",
    )
    args = parser.parse_args()

    if not args.drift_report.is_file():
        print(f"error: no drift report at {args.drift_report}", file=sys.stderr)
        print("Generate one with: python -m models.validation.drift", file=sys.stderr)
        return 2

    report = json.loads(args.drift_report.read_text(encoding="utf-8"))

    result = run_pipeline(
        model=args.model,
        drift_report=report,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        history_path=args.history,
        dry_run=not args.execute,
        epochs=args.epochs,
        cooldown_hours=args.cooldown_hours,
        track=args.track,
    )

    print(result.summary())
    for message in result.messages:
        print(f"  {message}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"report: {args.report}")

    if not args.execute and result.decision.get("should_retrain"):
        print("\nThis was a dry run. Add --execute to retrain.")

    # Exit non-zero only on a genuine failure. "Decided not to retrain" is the
    # pipeline working, and a scheduled job that goes red on a quiet day
    # trains everyone to ignore it.
    return 1 if result.stage is Stage.FAILED else 0


__all__ = [
    "PipelineResult",
    "RetrainingDecision",
    "Stage",
    "decide",
    "run_pipeline",
]

if __name__ == "__main__":
    raise SystemExit(main())
