"""The worker's lifecycle signal handlers.

These run once per worker process, outside any task, which is what makes them
easy to leave untested and awkward when they break: a failure here takes out
the whole process rather than one job, and Celery's own error reporting for
signal handlers is thin.

They are called directly. Going through Celery's signal dispatch would test
Celery.
"""

from __future__ import annotations

import os

import pytest


class TestWorkerProcessInit:
    """Runs in each forked worker before it accepts a job."""

    def test_it_pins_the_math_libraries_to_one_thread(self, monkeypatch) -> None:
        """Celery already runs one process per core. Leaving OpenMP and MKL
        to their own defaults means each of those processes also spawns a
        thread per core, and the resulting oversubscription makes inference
        slower than running single-threaded.
        """
        from worker.celery_app import _init_worker

        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

        _init_worker()

        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"

    def test_it_does_not_override_an_explicit_setting(self, monkeypatch) -> None:
        """setdefault, not assignment: an operator who tuned this deliberately
        should keep their value."""
        from worker.celery_app import _init_worker

        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        _init_worker()
        assert os.environ["OMP_NUM_THREADS"] == "4"


class TestWorkerProcessShutdown:
    """Runs as a worker process exits."""

    def test_it_releases_the_models(self, monkeypatch) -> None:
        from worker import celery_app as module

        unloaded: list[bool] = []

        class FakeService:
            def unload_all(self) -> None:
                unloaded.append(True)

        monkeypatch.setattr("api.services.model_service.get_model_service", lambda: FakeService())
        module._shutdown_worker()
        assert unloaded == [True]

    def test_a_failure_while_unloading_is_swallowed(self, monkeypatch) -> None:
        """Shutdown must never raise.

        An exception here happens while the process is already going away, so
        there is nothing useful it could achieve, and Celery reports it as a
        worker crash rather than a clean exit.
        """
        from worker import celery_app as module

        class Broken:
            def unload_all(self) -> None:
                raise RuntimeError("the GPU is already gone")

        monkeypatch.setattr("api.services.model_service.get_model_service", lambda: Broken())
        module._shutdown_worker()  # must not raise

    def test_it_survives_the_model_service_being_unimportable(self, monkeypatch) -> None:
        """At interpreter shutdown modules can already be torn down."""
        from worker import celery_app as module

        def explode():
            raise ImportError("module is being torn down")

        monkeypatch.setattr("api.services.model_service.get_model_service", explode)
        module._shutdown_worker()


class TestWorkerLogging:
    def test_the_worker_configures_the_same_logging_as_the_api(self, monkeypatch) -> None:
        """Otherwise worker output is unstructured and a correlation ID cannot
        be followed from a request into the job it queued."""
        from worker import celery_app as module

        calls: list[tuple] = []
        # Patch the name bound in this module, not in api.logging_config.
        # celery_app does `from api.logging_config import configure_logging`
        # at import time, so it holds its own reference and patching the
        # source module has no effect on it.
        monkeypatch.setattr(
            module, "configure_logging", lambda level, fmt: calls.append((level, fmt))
        )
        module._configure_worker_logging()

        assert len(calls) == 1
        level, fmt = calls[0]
        assert level and fmt


class TestTaskDiscovery:
    def test_the_app_reports_whether_tasks_are_importable(self) -> None:
        """The probe exists so a broker-only image, which deliberately omits
        the model stack, does not fail at import."""
        from worker import celery_app as module

        assert isinstance(module._TASKS_AVAILABLE, bool)

    def test_a_missing_tasks_module_is_not_fatal(self, monkeypatch) -> None:
        import importlib
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def no_tasks(name, *args, **kwargs):
            if name == "worker.tasks":
                raise ValueError("no such module")
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", no_tasks)

        from worker import celery_app as module

        reloaded = importlib.reload(module)
        assert reloaded._TASKS_AVAILABLE is False

        # Leave the module as the rest of the suite expects it.
        monkeypatch.undo()
        importlib.reload(module)


@pytest.mark.parametrize(
    "setting",
    ["task_acks_late", "worker_prefetch_multiplier", "task_track_started"],
)
def test_the_queue_is_configured_for_long_running_jobs(setting: str) -> None:
    """Inference jobs take seconds to minutes, not milliseconds.

    acks_late means a job survives a worker dying mid-run; a prefetch of 1
    stops one worker hoarding a queue of slow jobs while another idles.
    """
    from worker.celery_app import celery_app

    assert setting in celery_app.conf


class TestWorkerMetricsEndpoint:
    """Batch metrics are recorded in child processes; the parent serves them."""

    def test_nothing_starts_without_the_multiprocess_directory(self, monkeypatch) -> None:
        import prometheus_client

        from worker.celery_app import _serve_worker_metrics

        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        started: list[int] = []
        monkeypatch.setattr(
            prometheus_client, "start_http_server", lambda port, **_: started.append(port)
        )
        _serve_worker_metrics()
        assert started == []

    def test_serves_the_aggregate_and_clears_old_files(self, monkeypatch, tmp_path) -> None:
        import prometheus_client

        from worker.celery_app import _serve_worker_metrics

        directory = tmp_path / "prom"
        directory.mkdir()
        (directory / "counter_1234.db").write_bytes(b"left over from the last run")
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(directory))
        monkeypatch.setenv("WORKER_METRICS_PORT", "9808")
        calls: list[tuple] = []
        monkeypatch.setattr(
            prometheus_client,
            "start_http_server",
            lambda port, registry=None: calls.append((port, registry)),
        )
        _serve_worker_metrics()
        assert [port for port, _ in calls] == [9808]
        assert calls[0][1] is not None, "must serve a multiprocess registry, not the default"
        assert list(directory.iterdir()) == []
