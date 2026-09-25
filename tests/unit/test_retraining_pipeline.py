"""The retraining loop.

Almost all of these are about refusing. Training when asked is the easy half;
the half that protects production is knowing when not to, and never shipping a
model that failed a gate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from models.pipeline.retraining import (
    PipelineResult,
    Stage,
    decide,
    run_pipeline,
)


def drift_report(
    *,
    drifted: bool = True,
    severity: str = "high",
    features: list[str] | None = None,
    effect: float = 0.4,
) -> dict[str, Any]:
    features = features or ["confidence"]
    return {
        "model": "resnet50-tiny-imagenet",
        "generated_at": datetime.now(UTC).isoformat(),
        "overall_severity": severity,
        "drifted": drifted,
        "summary": "test",
        "results": [
            {
                "test": "ks",
                "feature": f,
                "drifted": True,
                "effect_size": effect,
                "severity": severity,
            }
            for f in features
        ],
    }


def history_entry(*, stage: Stage, hours_ago: float, drifted: bool, dry_run: bool = False) -> dict:
    return {
        "stage": stage.value,
        "dry_run": dry_run,
        "started_at": (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(),
        "decision": {"drifted_features": ["confidence"] if drifted else []},
    }


class TestWhenToRetrain:
    def test_high_severity_drift_triggers_a_retrain(self) -> None:
        decision = decide(drift_report(severity="high"))
        assert decision.should_retrain is True
        assert "confidence" in decision.reason

    def test_no_drift_means_no_retrain(self) -> None:
        decision = decide(drift_report(drifted=False, severity="none"))
        assert decision.should_retrain is False

    def test_low_severity_is_below_the_threshold(self) -> None:
        decision = decide(drift_report(severity="low"))
        assert decision.should_retrain is False


class TestNoiseSuppression:
    """Drift detectors are noisy. Acting on every signal makes the model
    worse, not better."""

    def test_a_significant_but_tiny_effect_is_ignored(self) -> None:
        """With enough samples everything is significant. Effect size is what
        separates a real shift from a large sample."""
        decision = decide(drift_report(severity="high", effect=0.01), min_effect_size=0.1)
        assert decision.should_retrain is False
        assert "effect-size" in decision.reason

    def test_moderate_drift_waits_to_be_seen_twice(self) -> None:
        decision = decide(drift_report(severity="moderate"), history=[])
        assert decision.should_retrain is False
        assert decision.blocked_by == "awaiting_confirmation"

    def test_moderate_drift_twice_running_does_trigger(self) -> None:
        history = [history_entry(stage=Stage.SKIPPED, hours_ago=25, drifted=True)]
        decision = decide(drift_report(severity="moderate"), history=history)
        assert decision.should_retrain is True
        assert "again" in decision.reason


class TestCooldown:
    """Drift persists for days. Without a cooldown the pipeline retrains on
    every run for as long as it lasts."""

    def test_a_recent_retrain_blocks_another(self) -> None:
        history = [history_entry(stage=Stage.PROMOTED, hours_ago=2, drifted=True)]
        decision = decide(drift_report(severity="high"), history=history, cooldown_hours=24)
        assert decision.should_retrain is False
        assert decision.blocked_by == "cooldown"

    def test_an_old_retrain_does_not(self) -> None:
        history = [history_entry(stage=Stage.PROMOTED, hours_ago=48, drifted=True)]
        decision = decide(drift_report(severity="high"), history=history, cooldown_hours=24)
        assert decision.should_retrain is True

    def test_a_skipped_run_does_not_start_a_cooldown(self) -> None:
        """Only actual training counts. Otherwise one skipped run suppresses
        the next 24 hours of real signals."""
        history = [history_entry(stage=Stage.SKIPPED, hours_ago=1, drifted=True)]
        decision = decide(drift_report(severity="high"), history=history, cooldown_hours=24)
        assert decision.should_retrain is True

    def test_a_dry_run_does_not_start_a_cooldown_either(self) -> None:
        history = [history_entry(stage=Stage.TRAINED, hours_ago=1, drifted=True, dry_run=True)]
        decision = decide(drift_report(severity="high"), history=history, cooldown_hours=24)
        assert decision.should_retrain is True


class TestTheGates:
    """Nothing promotes itself. Every stage after training can refuse, and a
    refusal leaves the current model serving."""

    @pytest.fixture
    def always_trains(self, monkeypatch):
        monkeypatch.setattr("models.pipeline.retraining._run_training", lambda **kw: True)

    def test_a_failed_training_run_promotes_nothing(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("models.pipeline.retraining._run_training", lambda **kw: False)
        result = _run(tmp_path, dry_run=False)
        assert result.stage is Stage.FAILED
        assert result.promoted is False

    def test_failing_validation_stops_the_promotion(
        self, monkeypatch, tmp_path, always_trains
    ) -> None:
        monkeypatch.setattr(
            "models.pipeline.retraining._run_validation",
            lambda m: {"passed": False, "critical_failures": ["output sanity"]},
        )
        result = _run(tmp_path, dry_run=False)
        assert result.stage is Stage.FAILED
        assert result.promoted is False

    def test_a_regression_against_the_baseline_stops_it_too(
        self, monkeypatch, tmp_path, always_trains
    ) -> None:
        """The model is internally sane but worse than the one already
        serving. Validation cannot catch this; only the baseline can."""
        monkeypatch.setattr(
            "models.pipeline.retraining._run_validation", lambda m: {"passed": True}
        )
        monkeypatch.setattr(
            "models.pipeline.retraining._run_regression",
            lambda m: {"passed": False, "failures": ["top1_accuracy dropped 3.2 points"]},
        )
        result = _run(tmp_path, dry_run=False)
        assert result.stage is Stage.FAILED
        assert result.promoted is False
        assert "top1_accuracy" in " ".join(result.messages)

    def test_passing_both_gates_promotes(self, monkeypatch, tmp_path, always_trains) -> None:
        monkeypatch.setattr(
            "models.pipeline.retraining._run_validation", lambda m: {"passed": True}
        )
        monkeypatch.setattr(
            "models.pipeline.retraining._run_regression", lambda m: {"passed": True}
        )
        result = _run(tmp_path, dry_run=False)
        assert result.stage is Stage.PROMOTED
        assert result.promoted is True

    def test_no_baseline_yet_is_not_a_regression(
        self, monkeypatch, tmp_path, always_trains
    ) -> None:
        """The first model has nothing to regress against."""
        monkeypatch.setattr(
            "models.pipeline.retraining._run_validation", lambda m: {"passed": True}
        )
        monkeypatch.setattr("models.pipeline.retraining._run_regression", lambda m: None)
        result = _run(tmp_path, dry_run=False)
        assert result.stage is Stage.PROMOTED


class TestDryRun:
    def test_the_default_trains_nothing(self, monkeypatch, tmp_path) -> None:
        def explode(**kwargs):
            raise AssertionError("a dry run must not start training")

        monkeypatch.setattr("models.pipeline.retraining._run_training", explode)
        result = _run(tmp_path, dry_run=True)
        assert result.stage is Stage.ASSESSED
        assert result.promoted is False
        assert any("dry run" in m for m in result.messages)

    def test_it_still_reports_the_decision(self, tmp_path) -> None:
        result = _run(tmp_path, dry_run=True)
        assert result.decision["should_retrain"] is True


class TestHistory:
    def test_every_run_is_recorded(self, tmp_path) -> None:
        path = tmp_path / "history.json"
        _run(tmp_path, dry_run=True, history_path=path)
        _run(tmp_path, dry_run=True, history_path=path)
        assert len(json.loads(path.read_text(encoding="utf-8"))) == 2

    def test_a_corrupt_history_does_not_kill_the_run(self, tmp_path) -> None:
        """This runs on a schedule. Crashing on a bad file means nobody finds
        out until someone wonders why drift stopped being reported."""
        path = tmp_path / "history.json"
        path.write_text("{ not json", encoding="utf-8")
        result = _run(tmp_path, dry_run=True, history_path=path)
        assert result.stage in (Stage.ASSESSED, Stage.SKIPPED)

    def test_the_skip_reason_is_recorded_not_just_the_skip(self, tmp_path) -> None:
        """A 'no' has to be auditable, or the next person cannot tell a
        working policy from a broken detector."""
        path = tmp_path / "history.json"
        _run(tmp_path, dry_run=True, history_path=path, report=drift_report(severity="low"))
        entry = json.loads(path.read_text(encoding="utf-8"))[0]
        assert entry["stage"] == Stage.SKIPPED.value
        assert entry["decision"]["reason"]


def _run(
    tmp_path: Path,
    *,
    dry_run: bool,
    history_path: Path | None = None,
    report: dict[str, Any] | None = None,
) -> PipelineResult:
    return run_pipeline(
        model="resnet50-tiny-imagenet",
        drift_report=report or drift_report(),
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        history_path=history_path,
        dry_run=dry_run,
        epochs=1,
        track="none",
    )


class TestTheTrainingCommand:
    """The one seam the mocked gates cannot cover.

    `_run_training` builds a command line and hands it to a subprocess. If a
    flag is renamed in the training CLI, every other test here still passes
    and the pipeline fails only when someone runs it for real, an hour into a
    job that was supposed to retrain.
    """

    def test_the_command_is_accepted_by_the_training_cli(self, monkeypatch, tmp_path) -> None:
        captured: dict[str, list[str]] = {}

        class Completed:
            returncode = 0

        def capture(command, **kwargs):
            captured["command"] = command
            return Completed()

        monkeypatch.setattr("models.pipeline.retraining.subprocess.run", capture)

        from models.pipeline.retraining import _run_training

        assert _run_training(epochs=3, data_dir=tmp_path, output_dir=tmp_path, track="none")

        command = captured["command"]
        assert command[1:3] == ["-m", "models.training.train_classifier"]

        # Run the real parser over the real flags. This fails if any of them
        # is renamed or dropped.
        import subprocess

        done = subprocess.run([*command, "--help"], capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, (
            "the training CLI rejected the pipeline's arguments:\n" + done.stderr[-800:]
        )
