"""Celery application for background batch processing.

Plain English:
    Some requests take too long to answer inside an HTTP call. Classifying 64
    images might take a minute; browsers, proxies and load balancers will all
    give up long before that.

    So the API does not do the work. It writes the job onto a queue (Redis),
    returns a job id immediately, and a separate *worker* process picks the
    job up and does the work. The caller polls for the result.

    This also means the two halves scale independently: add API containers for
    more concurrent connections, add worker containers for more throughput.

Configuration choices that matter in production:

* ``task_acks_late=True`` — a task is only acknowledged *after* it finishes.
  If a worker is killed mid-task, the job returns to the queue instead of
  vanishing.
* ``worker_prefetch_multiplier=1`` — each worker reserves one task at a time.
  The default of 4 lets a single worker hoard tasks while others idle, which
  is exactly wrong for long, uneven ML jobs.
* ``worker_max_tasks_per_child`` — restart a worker process after N tasks.
  This is a blunt but effective defence against the slow memory growth that
  ML libraries are prone to.
"""

from __future__ import annotations

import os
from pathlib import Path

from celery import Celery
from celery.signals import (
    setup_logging,
    worker_init,
    worker_process_init,
    worker_process_shutdown,
)

from api.config import settings
from api.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

# prometheus_client in multiprocess mode fails on the first metric write if
# this directory does not exist, and a tmpfs /tmp starts empty.
if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
    Path(os.environ["PROMETHEUS_MULTIPROC_DIR"]).mkdir(parents=True, exist_ok=True)

# The task module is registered only when it can actually be imported.
#
# The same Celery app object is used by two very different processes:
#   * the worker, which needs the task implementations, and
#   * the API, which only enqueues by name (`send_task`) and reads results.
#
# The API image deliberately does not ship worker/tasks.py — it would pull the
# whole model-loading stack in for no reason — so an unconditional
# `include=["worker.tasks"]` would raise ModuleNotFoundError there.
try:  # pragma: no cover - depends on which image this runs in
    import importlib.util

    _TASKS_AVAILABLE = importlib.util.find_spec("worker.tasks") is not None
except (ImportError, ValueError):
    _TASKS_AVAILABLE = False

celery_app = Celery(
    "cv_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["worker.tasks"] if _TASKS_AVAILABLE else [],
)

celery_app.conf.update(
    # --- Serialisation -----------------------------------------------------
    # JSON only. Celery's default used to be pickle, which executes arbitrary
    # code on deserialisation — anyone who can write to the queue gets remote
    # code execution on every worker.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # --- Reliability -------------------------------------------------------
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    # --- Timeouts ----------------------------------------------------------
    # The soft limit raises an exception the task can catch and report;
    # the hard limit kills the process. Keeping them apart means most
    # overruns produce a useful error instead of a silent disappearance.
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    # --- Results -----------------------------------------------------------
    result_expires=86400,  # keep results for 24 h, then let Redis reclaim them
    result_extended=True,  # store task name and args, useful when debugging
    # --- Routing -----------------------------------------------------------
    task_default_queue="inference",
    task_queues_late_ack=True,
    timezone="UTC",
    enable_utc=True,
    # --- Visibility --------------------------------------------------------
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_track_started=True,
)


@setup_logging.connect
def _configure_worker_logging(**_: object) -> None:
    """Use our JSON logging in the worker too.

    Without this, Celery installs its own formatter and worker logs come out
    in a different shape from API logs — which makes the two impossible to
    correlate in a log aggregator.
    """
    configure_logging(settings.log_level, settings.log_format)


@worker_init.connect
def _serve_worker_metrics(**_: object) -> None:
    """Serve the worker's Prometheus metrics, summed across child processes.

    batch_jobs_total and batch_job_duration_seconds are recorded where the
    batch runs, in a Celery child process that nothing scrapes. With
    PROMETHEUS_MULTIPROC_DIR set, each child writes to that directory and this
    endpoint, started once in the parent before it forks, reads them all.
    Without the variable (tests, local runs) nothing is started.
    """
    directory = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    port = os.getenv("WORKER_METRICS_PORT")
    if not directory or not port:
        return
    import shutil

    from prometheus_client import CollectorRegistry, multiprocess, start_http_server

    # Files left by a previous run would be summed into this one's counters.
    shutil.rmtree(directory, ignore_errors=True)
    Path(directory).mkdir(parents=True, exist_ok=True)
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    start_http_server(int(port), registry=registry)
    logger.info("worker_metrics_serving", extra={"port": int(port)})


@worker_process_init.connect
def _init_worker(**_: object) -> None:
    """Prepare each worker process when it starts.

    Models are loaded per worker process, not per task. Loading a model takes
    seconds; doing it for every batch would dwarf the actual work.

    Thread counts are pinned to 1 because Celery already runs several worker
    processes. Letting each of them spawn a thread per core causes massive
    oversubscription, where the CPU spends its time context-switching instead
    of computing.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    logger.info("worker_process_started", extra={"pid": os.getpid()})


@worker_process_shutdown.connect
def _shutdown_worker(**_: object) -> None:
    """Release models and connections when a worker process exits."""
    from api.services.model_service import get_model_service

    try:
        get_model_service().unload_all()
    except Exception:  # shutdown must never raise
        pass
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        try:
            from prometheus_client import multiprocess

            # Otherwise the recycled process's live gauges keep being summed.
            multiprocess.mark_process_dead(os.getpid())
        except Exception:
            pass
    logger.info("worker_process_stopped", extra={"pid": os.getpid()})


__all__ = ["celery_app"]
