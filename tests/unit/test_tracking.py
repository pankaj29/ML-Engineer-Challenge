"""Experiment tracking.

The property worth testing is not that metrics arrive. It is that a broken
tracking backend cannot take a training run with it. A run costs hours on a
GPU; a metrics sink is a convenience, and the failure modes must not be
symmetrical.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.training.tracking import MLflowTracker, NullTracker, get_tracker


class TestSelection:
    def test_none_gives_the_no_op(self) -> None:
        assert isinstance(get_tracker("none"), NullTracker)

    def test_the_default_is_off(self) -> None:
        """Opt-in. A run should not write into a tracking store uninvited."""
        assert isinstance(get_tracker(), NullTracker)

    def test_auto_never_raises_whatever_is_installed(self) -> None:
        tracker = get_tracker("auto")
        assert hasattr(tracker, "log_metrics")

    def test_auto_falls_back_when_mlflow_is_missing(self, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def no_mlflow(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("no mlflow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_mlflow)
        assert isinstance(get_tracker("auto"), NullTracker)

    def test_an_unknown_mode_is_refused(self) -> None:
        """Rather than silently tracking nothing, which is indistinguishable
        from a typo in a flag."""
        with pytest.raises(ValueError, match="unknown tracking mode"):
            get_tracker("wandb")


class TestTheNoOpAcceptsEverything:
    """The training loop calls these unconditionally, so none may raise."""

    def test_every_method_is_safe_to_call(self, tmp_path: Path) -> None:
        tracker = NullTracker()
        tracker.start({"lr": 0.001}, run_name="x")
        tracker.log_metrics({"loss": 1.0}, step=1)
        tracker.log_artifact(tmp_path / "missing.json")
        tracker.set_summary({"best": 78.9})
        tracker.finish()


class TestFailuresAreContained:
    """A tracking backend that breaks mid-run must not end the run."""

    @staticmethod
    def _broken() -> MLflowTracker:
        class Exploding:
            def __getattr__(self, name):
                def boom(*args, **kwargs):
                    raise RuntimeError(f"{name} is down")

                return boom

        tracker = MLflowTracker.__new__(MLflowTracker)
        tracker.enabled = True
        tracker._mlflow = Exploding()
        tracker._experiment = "test"
        tracker._tracking_uri = None
        return tracker

    def test_a_failing_start_does_not_raise(self) -> None:
        tracker = self._broken()
        tracker.start({"a": 1})
        assert tracker.enabled is False, "tracking should switch itself off after failing"

    def test_it_stops_trying_after_the_first_failure(self, caplog) -> None:
        """Otherwise a dead backend produces one warning per epoch, and the
        real training output scrolls away."""
        tracker = self._broken()
        with caplog.at_level("WARNING"):
            tracker.start({"a": 1})
            for step in range(20):
                tracker.log_metrics({"loss": 0.1}, step=step)
            tracker.finish()

        warnings = [r for r in caplog.records if "tracking disabled" in r.message]
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"

    def test_a_missing_artifact_is_skipped_not_raised(self, tmp_path: Path) -> None:
        tracker = self._broken()
        tracker.log_artifact(tmp_path / "nope.json")
        assert tracker.enabled is True, "a missing file should not disable tracking"


class TestMetricFiltering:
    def test_non_numeric_metrics_are_dropped(self) -> None:
        """MLflow rejects the whole call on a bad value, which would lose the
        good metrics alongside the bad one."""
        recorded: dict = {}

        class Recorder:
            def log_metrics(self, metrics, step):
                recorded.update(metrics)

        tracker = MLflowTracker.__new__(MLflowTracker)
        tracker.enabled = True
        tracker._mlflow = Recorder()

        tracker.log_metrics({"top1": 78.9, "weights": "ema", "ema_top1": None}, step=3)
        assert recorded == {"top1": 78.9}


@pytest.mark.integration
class TestAgainstRealMLflow:
    """Runs only where mlflow is installed. Writes to a temp directory."""

    def test_a_run_is_recorded_and_readable(self, tmp_path: Path) -> None:
        mlflow = pytest.importorskip("mlflow")

        uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
        tracker = get_tracker("mlflow", experiment="unit-test", tracking_uri=uri)
        if isinstance(tracker, NullTracker):
            pytest.skip("mlflow present but unusable here")

        tracker.start({"arch": "resnet50", "epochs": 2}, run_name="probe")
        tracker.log_metrics({"val_top1": 70.0}, step=1)
        tracker.log_metrics({"val_top1": 78.9}, step=2)
        tracker.set_summary({"best_top1": 78.9})
        tracker.finish()

        mlflow.set_tracking_uri(uri)
        runs = mlflow.search_runs(experiment_names=["unit-test"])
        assert len(runs) == 1

        row = runs.iloc[0]
        assert row["params.arch"] == "resnet50"
        assert row["metrics.val_top1"] == pytest.approx(78.9), "should hold the last step"
        assert row["tags.best_top1"] == "78.9"
