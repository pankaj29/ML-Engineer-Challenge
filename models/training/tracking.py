"""Experiment tracking for training runs.

Plain English:
    Every run already writes a history JSON next to its checkpoint. That is
    enough to plot one run and useless for comparing twenty: the files have
    the same name, sit in the same directory, and the later run overwrites the
    earlier one. Tracking gives each run an id, records what it was configured
    with, and keeps the metrics somewhere they can be compared.

MLflow, not Weights and Biases, and not TensorBoard. It runs against a local
SQLite file, so it needs no account, no network and no server, which matters
because most of these runs happen on a Colab box that is deleted afterwards.
Pass ``--tracking-uri`` to log to a shared server instead.

The default is SQLite rather than the older ``./mlruns`` directory because
current MLflow refuses the file store outright: it raises *"The filesystem
tracking backend is in maintenance mode"* unless ``MLFLOW_ALLOW_FILE_STORE``
is set. SQLite is the supported local option and needs nothing extra.

Tracking is optional and never fatal. If mlflow is not installed, or the
tracking store cannot be reached, training continues and says so once. Losing
a metrics sink is not a reason to lose a two-hour GPU run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: Local store used when no tracking URI is given. A file, not a server, and
#: not the deprecated ./mlruns directory.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


class Tracker(Protocol):
    """What the training loop needs from a tracking backend."""

    def start(self, params: Mapping[str, Any], run_name: str | None = None) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def set_summary(self, values: Mapping[str, Any]) -> None: ...
    def finish(self, status: str = "FINISHED") -> None: ...


class NullTracker:
    """Does nothing, so the training loop needs no conditionals.

    This is the default. Tracking is opt-in because a run that silently starts
    writing into someone's MLflow store is a surprise, not a feature.
    """

    enabled = False

    def start(self, params: Mapping[str, Any], run_name: str | None = None) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        return None

    def log_artifact(self, path: Path) -> None:
        return None

    def set_summary(self, values: Mapping[str, Any]) -> None:
        return None

    def finish(self, status: str = "FINISHED") -> None:
        return None


class MLflowTracker:
    """Logs to MLflow, degrading to a no-op on any failure.

    Every method swallows its exceptions. That is usually the wrong instinct,
    but the alternative here is a network blip in the metrics sink killing a
    training run that is otherwise fine. The first failure is logged and
    tracking switches itself off, so the log carries one warning rather than
    one per epoch.
    """

    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        self.enabled = False
        self._mlflow: Any = None
        self._experiment = experiment
        self._tracking_uri = tracking_uri

        try:
            import mlflow
        except ImportError:
            logger.warning(
                "mlflow is not installed, so this run will not be tracked. "
                "pip install mlflow, or pass --track none to silence this."
            )
            return

        self._mlflow = mlflow
        self.enabled = True

    def _give_up(self, what: str, exc: Exception) -> None:
        logger.warning(
            "experiment tracking disabled after %s failed: %s: %s",
            what,
            type(exc).__name__,
            exc,
        )
        self.enabled = False

    def start(self, params: Mapping[str, Any], run_name: str | None = None) -> None:
        if not self.enabled:
            return
        try:
            if self._tracking_uri:
                self._mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow.set_experiment(self._experiment)
            self._mlflow.start_run(run_name=run_name)
            # MLflow stores parameters as strings and rejects values over 500
            # characters, so anything long is truncated rather than failing
            # the whole call and losing the other parameters with it.
            self._mlflow.log_params({k: str(v)[:500] for k, v in params.items() if v is not None})
            logger.info("tracking run started in experiment %r", self._experiment)
        except Exception as exc:
            self._give_up("start_run", exc)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        if not self.enabled:
            return
        try:
            clean = {
                k: float(v)
                for k, v in metrics.items()
                if v is not None and isinstance(v, (int, float))
            }
            self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:
            self._give_up("log_metrics", exc)

    def log_artifact(self, path: Path) -> None:
        if not self.enabled or not Path(path).exists():
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            self._give_up("log_artifact", exc)

    def set_summary(self, values: Mapping[str, Any]) -> None:
        """Record the run's headline numbers as tags.

        Tags rather than metrics because these are the values you filter a run
        list by, and MLflow's UI filters on tags.
        """
        if not self.enabled:
            return
        try:
            self._mlflow.set_tags({k: str(v) for k, v in values.items() if v is not None})
        except Exception as exc:
            self._give_up("set_tags", exc)

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:
            self._give_up("end_run", exc)


def get_tracker(
    mode: str = "none",
    *,
    experiment: str = "tiny-imagenet-classification",
    tracking_uri: str | None = None,
) -> Tracker:
    """Build the tracker named by ``mode``.

    ``"none"`` gives the no-op, ``"mlflow"`` gives MLflow, and ``"auto"`` uses
    MLflow when it is importable and the no-op otherwise. ``auto`` exists for
    the Colab notebook, which cannot know in advance what the runtime has.
    """
    mode = (mode or "none").lower()

    if mode == "none":
        return NullTracker()

    if mode == "auto":
        try:
            import mlflow  # noqa: F401
        except ImportError:
            logger.info("mlflow not available, continuing without tracking")
            return NullTracker()
        mode = "mlflow"

    if mode == "mlflow":
        tracker = MLflowTracker(
            experiment=experiment,
            tracking_uri=tracking_uri or DEFAULT_TRACKING_URI,
        )
        return tracker if tracker.enabled else NullTracker()

    raise ValueError(f"unknown tracking mode {mode!r}: expected none, auto or mlflow")


__all__ = ["DEFAULT_TRACKING_URI", "MLflowTracker", "NullTracker", "Tracker", "get_tracker"]
