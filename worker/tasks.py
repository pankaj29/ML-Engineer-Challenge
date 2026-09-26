"""Celery tasks for batch inference.

Plain English:
    This is the code that actually runs when a batch job is picked off the
    queue. It walks the submitted images one at a time, runs the requested
    model, and records the outcome for each.

The rule that shapes everything here: **one bad image must not fail the
batch**. If image 17 of 64 is corrupt, images 1-16 and 18-64 still return
their results, and item 17 carries an error object explaining why. A caller
who submits 64 images and gets a single top-level error learns nothing about
which one was the problem.

Progress is written back after every item, so ``GET /batch/{job_id}`` shows a
live percentage rather than jumping from 0% to 100%.
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from api.exceptions import AppError
from api.logging_config import bind_correlation_id, get_logger
from api.models.schemas import TaskType
from worker.celery_app import celery_app

logger = get_logger(__name__)


class InferenceTask(Task):
    """Base task that keeps models loaded between jobs.

    Celery creates one instance of a task class per worker process and reuses
    it, so anything cached on ``self`` survives across tasks in that process.
    That is exactly what we want for models, which are expensive to load and
    cheap to reuse.
    """

    _services: dict[str, Any] | None = None

    @property
    def services(self) -> dict[str, Any]:
        """Lazily build and cache the service objects for this worker process."""
        if self._services is None:
            from api.services.cache_service import CacheService
            from api.services.inference_service import InferenceService
            from api.services.model_service import ModelService

            model_service = ModelService()
            cache_service = CacheService()
            self._services = {
                "models": model_service,
                "cache": cache_service,
                "inference": InferenceService(model_service, cache_service),
            }
            logger.info("worker_services_initialised")
        return self._services


@lru_cache(maxsize=1)
def _job_engine() -> Any:
    """Sync engine for the batch_jobs table, one per worker process.

    The worker runs a fresh event loop per batch, and an asyncpg pool is bound
    to the loop that created it, so this uses the sync psycopg driver instead.
    """
    from sqlalchemy import create_engine

    from api.config import settings

    url = settings.database_url.replace("+asyncpg", "+psycopg").replace("+aiosqlite", "")
    return create_engine(url, pool_pre_ping=True)


def _record_job(job_id: str, **fields: Any) -> None:
    """Write job state to batch_jobs. Best effort: never fails the batch.

    Celery's result expires after 24 hours; this row is what the status
    endpoint falls back to afterwards.
    """
    try:
        from sqlalchemy import update

        from db.models import BatchJob

        with _job_engine().begin() as conn:
            conn.execute(update(BatchJob).where(BatchJob.id == job_id).values(**fields))
    except Exception as exc:
        logger.warning(
            "batch_job_record_failed",
            extra={"job_id": job_id, "error": f"{type(exc).__name__}: {exc}"},
        )


def _decode_item(item: dict[str, Any]) -> bytes:
    """Turn one submitted batch item into image bytes.

    Raises:
        ValueError: The item carries neither usable base64 nor a URL.
    """
    if item.get("image_base64"):
        return base64.b64decode(item["image_base64"], validate=True)
    if item.get("image_url"):
        import httpx

        from api.config import settings
        from api.exceptions import ImageTooLargeError
        from api.utils.validators import validate_image_url

        url = validate_image_url(item["image_url"])
        limit = settings.max_image_bytes
        # Streamed with a cap, like the API's fetch: reading response.content
        # would pull an arbitrarily large body into worker memory first.
        with (
            httpx.Client(timeout=10.0, follow_redirects=False) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes(64 * 1024):
                total += len(chunk)
                if total > limit:
                    raise ImageTooLargeError(
                        f"The remote image exceeds the {limit / 1_048_576:.0f} MB limit.",
                        details={"limit_bytes": limit},
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    raise ValueError("item contains neither image_base64 nor image_url")


async def _run_one(
    inference: Any,
    task_type: TaskType,
    image_bytes: bytes,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run one image through the requested model and return a plain dict."""
    if task_type == TaskType.CLASSIFICATION:
        result = await inference.classify(
            image_bytes,
            top_k=params.get("top_k", 5),
            confidence_threshold=params.get("confidence_threshold", 0.0),
            model_name=params.get("model_name"),
            model_version=params.get("model_version"),
        )
    elif task_type == TaskType.DETECTION:
        result = await inference.detect(
            image_bytes,
            confidence_threshold=params.get("confidence_threshold", 0.25),
            iou_threshold=params.get("iou_threshold", 0.45),
            max_detections=params.get("max_detections", 100),
            model_name=params.get("model_name"),
            model_version=params.get("model_version"),
        )
    elif task_type == TaskType.SIMILARITY:
        result = await inference.embed(
            image_bytes,
            model_name=params.get("model_name"),
            model_version=params.get("model_version"),
        )
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    return result.model_dump(mode="json")


@celery_app.task(
    bind=True,
    base=InferenceTask,
    name="worker.tasks.process_batch",
    # Retry only on infrastructure failures, never on a bad image: retrying a
    # corrupt JPEG three times just wastes three times the CPU.
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=60,
    max_retries=3,
)
def process_batch(
    self: InferenceTask,
    job_id: str,
    task_type: str,
    items: list[dict[str, Any]],
    params: dict[str, Any],
    correlation_id: str = "",
) -> dict[str, Any]:
    """Process a batch of images.

    Args:
        job_id: Identifier for this job, matching the Celery task id.
        task_type: ``"classification"``, ``"detection"`` or ``"similarity"``.
        items: The submitted images.
        params: Task parameters (``top_k``, thresholds, model pinning).
        correlation_id: Propagated from the API request, so the whole journey
            through API, queue and worker shares one traceable id.

    Returns:
        A summary dict with per-item results, stored as the Celery result.
    """
    bind_correlation_id(correlation_id or None)
    started_at = datetime.now(UTC)
    started_perf = time.perf_counter()

    logger.info(
        "batch_job_started",
        extra={"job_id": job_id, "task": task_type, "items": len(items)},
    )
    _record_job(job_id, status="running", started_at=started_at)

    inference = self.services["inference"]
    ttype = TaskType(task_type)

    results: list[dict[str, Any]] = []
    completed = 0
    failed = 0

    # One event loop for the whole batch. Creating a loop per image would add
    # meaningful overhead across 64 items.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        for idx, item in enumerate(items):
            item_started = time.perf_counter()
            try:
                image_bytes = _decode_item(item)

                from api.utils.validators import validate_image_bytes

                validate_image_bytes(image_bytes, field_name=f"items[{idx}]")

                payload = loop.run_until_complete(_run_one(inference, ttype, image_bytes, params))
                results.append(
                    {
                        "index": idx,
                        "image_id": item.get("image_id"),
                        "success": True,
                        "result": payload,
                        "error": None,
                        "duration_ms": round((time.perf_counter() - item_started) * 1000, 2),
                    }
                )
                completed += 1

            except SoftTimeLimitExceeded:
                # The whole job is out of time. Stop, but keep what we have:
                # partial results are far more useful than none.
                logger.error(
                    "batch_job_soft_timeout",
                    extra={"job_id": job_id, "completed": completed, "remaining": len(items) - idx},
                )
                results.append(
                    {
                        "index": idx,
                        "image_id": item.get("image_id"),
                        "success": False,
                        "result": None,
                        "error": {
                            "code": "JOB_TIMEOUT",
                            "message": "The job exceeded its time limit before reaching this item.",
                        },
                        "duration_ms": None,
                    }
                )
                failed += 1
                break

            except AppError as exc:
                results.append(
                    {
                        "index": idx,
                        "image_id": item.get("image_id"),
                        "success": False,
                        "result": None,
                        "error": {"code": exc.code, "message": exc.message, "details": exc.details},
                        "duration_ms": round((time.perf_counter() - item_started) * 1000, 2),
                    }
                )
                failed += 1

            except Exception as exc:
                logger.warning(
                    "batch_item_failed",
                    extra={"job_id": job_id, "index": idx, "error": f"{type(exc).__name__}: {exc}"},
                )
                results.append(
                    {
                        "index": idx,
                        "image_id": item.get("image_id"),
                        "success": False,
                        "result": None,
                        "error": {
                            "code": "ITEM_PROCESSING_FAILED",
                            "message": "This image could not be processed.",
                        },
                        "duration_ms": round((time.perf_counter() - item_started) * 1000, 2),
                    }
                )
                failed += 1

            # Publish progress so the status endpoint can show a live figure.
            self.update_state(
                state="PROGRESS",
                meta={
                    "job_id": job_id,
                    "completed": completed,
                    "failed": failed,
                    "total": len(items),
                    "progress_percent": round((completed + failed) / len(items) * 100, 2),
                },
            )
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    duration = time.perf_counter() - started_perf
    completed_at = datetime.now(UTC)

    summary = {
        "job_id": job_id,
        "task": task_type,
        "status": "completed" if failed == 0 else ("failed" if completed == 0 else "completed"),
        "total_items": len(items),
        "completed_items": completed,
        "failed_items": failed,
        "results": results,
        "submitted_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round(duration, 2),
        "correlation_id": correlation_id,
    }

    _record_job(
        job_id,
        status=summary["status"],
        completed_items=completed,
        failed_items=failed,
        completed_at=completed_at,
    )

    from api.middleware.monitoring import batch_job_duration_seconds, batch_jobs_total

    batch_jobs_total.labels(task=task_type, status=summary["status"]).inc()
    batch_job_duration_seconds.labels(task=task_type).observe(duration)

    logger.info(
        "batch_job_finished",
        extra={
            "job_id": job_id,
            "completed": completed,
            "failed": failed,
            "duration_s": round(duration, 2),
            "per_image_ms": round(duration * 1000 / max(len(items), 1), 1),
        },
    )

    if params.get("callback_url"):
        _post_callback(params["callback_url"], summary)

    return summary


def _post_callback(url: str, payload: dict[str, Any]) -> None:
    """POST the finished result to a caller-supplied URL.

    Best-effort: a callback that fails must not fail the job, whose real
    results are already stored and retrievable by polling.
    """
    try:
        import httpx

        from api.utils.validators import validate_image_url

        validate_image_url(url)  # same SSRF protections as image fetching
        with httpx.Client(timeout=10.0, follow_redirects=False) as client:
            response = client.post(url, json=payload)
        logger.info(
            "batch_callback_sent",
            extra={"status_code": response.status_code, "job_id": payload.get("job_id")},
        )
    except Exception as exc:
        logger.warning(
            "batch_callback_failed",
            extra={"job_id": payload.get("job_id"), "error": f"{type(exc).__name__}: {exc}"},
        )


@celery_app.task(name="worker.tasks.health_check")
def health_check() -> dict[str, Any]:
    """Trivial task used to prove the worker is reachable and processing."""
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


__all__ = ["health_check", "process_batch"]
