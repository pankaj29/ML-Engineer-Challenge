"""Batch processing endpoints.

Plain English:
    Sending 64 images and waiting for all of them inside one HTTP request is a
    bad idea: browsers, proxies and load balancers all time out long before a
    big batch finishes.

    So this endpoint **accepts** the work and returns immediately with HTTP
    202 ("Accepted") and a job id. A Celery worker does the actual processing,
    and the caller polls ``GET /batch/{job_id}`` for progress and results.

Batch requests cost more than one rate-limit token. A 50-image batch is fifty
times the work of one image, and charging it as a single request would let a
caller bypass their limit entirely by wrapping everything in batches.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Path, Query, status

from api.dependencies import CorrelationId, CurrentPrincipal, Limiter, Models
from api.exceptions import JobNotFoundError, ServiceUnavailableError
from api.logging_config import get_logger
from api.models.responses import (
    BatchItemResult,
    BatchStatusResponse,
    BatchSubmitResponse,
    ErrorResponse,
)
from api.models.schemas import BatchRequest, JobStatus, TaskType
from api.utils.validators import validate_batch_size

if TYPE_CHECKING:
    from api.middleware.auth import Principal
    from db.models import BatchJob

logger = get_logger(__name__)

router = APIRouter(prefix="/batch", tags=["Batch"])

# Rough per-image cost used only for the completion estimate returned at
# submission. Refined by real measurements in the benchmark reports.
_ESTIMATED_SECONDS_PER_IMAGE = {
    TaskType.CLASSIFICATION: 0.12,
    TaskType.DETECTION: 0.35,
    TaskType.SIMILARITY: 0.12,
}


@router.post(
    "",
    response_model=BatchSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch of images for background processing",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid credentials."},
        413: {"model": ErrorResponse, "description": "Batch exceeds your tier's limit."},
        422: {"model": ErrorResponse, "description": "The request body failed validation."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded for your tier."},
        503: {"model": ErrorResponse, "description": "The job queue is unavailable."},
    },
)
async def submit_batch(
    body: BatchRequest,
    principal: CurrentPrincipal,
    limiter: Limiter,
    models: Models,
    correlation_id: CorrelationId,
) -> BatchSubmitResponse:
    """Queue a batch of images and return a job id immediately.

    The response is HTTP 202, meaning "accepted for processing" — not
    "finished". Poll ``status_url`` until ``status`` becomes ``completed`` or
    ``failed``.

    Per-tier batch caps: free 5, basic 16, pro 32, enterprise 64.

    Example response::

        {
          "job_id": "9f2c8a1e-...",
          "status": "pending",
          "total_items": 20,
          "status_url": "/api/v1/batch/9f2c8a1e-...",
          "estimated_seconds": 2.4
        }
    """
    validate_batch_size(len(body.items), principal.tier.value)

    # Charge the rate limiter proportionally to the work requested.
    await limiter.enforce(principal, cost=len(body.items))

    # Fail early with a clear message if no model exists for this task, rather
    # than queueing work that is certain to fail in the worker.
    entry = models.resolve(body.task, body.model_name, body.model_version)

    params = {
        "top_k": body.top_k,
        "confidence_threshold": body.confidence_threshold,
        "iou_threshold": body.iou_threshold,
        "max_detections": body.max_detections,
        "model_name": entry.name,
        "model_version": entry.version,
        "callback_url": body.callback_url,
    }
    items = [item.model_dump(mode="json") for item in body.items]

    try:
        import uuid

        from worker.celery_app import celery_app

        # The job id is generated here rather than letting Celery assign one,
        # so the same id is written to Postgres, sent to the worker and
        # returned to the caller — one identifier across all three.
        job_id = str(uuid.uuid4())

        # `send_task` addresses the task by NAME rather than importing it.
        # That keeps the API image free of the worker's implementation (and
        # its model-loading imports): the API needs only the broker
        # configuration to put a message on the queue. Importing the task
        # module here is what made the API image require the whole worker
        # package, and failed with ModuleNotFoundError when it did not.
        celery_app.send_task(
            "worker.tasks.process_batch",
            kwargs={
                "job_id": job_id,
                "task_type": body.task.value,
                "items": items,
                "params": params,
                "correlation_id": correlation_id,
            },
            task_id=job_id,
            queue="inference",
            priority=body.priority,
        )
    except Exception as exc:
        logger.error("batch_enqueue_failed", extra={"error": f"{type(exc).__name__}: {exc}"})
        raise ServiceUnavailableError(
            "The background job queue is unavailable. Try again shortly, or "
            "use the single-image endpoints.",
            internal_message=f"celery enqueue failed: {type(exc).__name__}: {exc}",
        ) from exc

    submitted_at = datetime.now(UTC)

    # Record the job so status survives a worker restart.
    from api.services.db_service import get_db_service

    await get_db_service().create_batch_job(
        {
            "id": job_id,
            "correlation_id": correlation_id,
            "user_id": principal.user_id,
            "user_tier": principal.tier.value,
            "task": body.task.value,
            "model_name": entry.name,
            "model_version": entry.version,
            "status": JobStatus.PENDING.value,
            "total_items": len(items),
            "params": params,
            "callback_url": body.callback_url,
            "submitted_at": submitted_at,
        }
    )

    logger.info(
        "batch_submitted",
        extra={
            "job_id": job_id,
            "task": body.task.value,
            "items": len(items),
            "tier": principal.tier.value,
        },
    )

    per_image = _ESTIMATED_SECONDS_PER_IMAGE.get(body.task, 0.2)
    return BatchSubmitResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        task=body.task,
        total_items=len(items),
        status_url=f"/api/v1/batch/{job_id}",
        estimated_seconds=round(len(items) * per_image, 1),
        correlation_id=correlation_id,
        submitted_at=submitted_at,
    )


@router.get(
    "/{job_id}",
    response_model=BatchStatusResponse,
    summary="Get batch job status and results",
    responses={
        404: {"model": ErrorResponse, "description": "No job with that id."},
    },
)
async def get_batch_status(
    principal: CurrentPrincipal,
    job_id: str = Path(description="Job id returned when the batch was submitted."),
    include_results: bool = Query(
        default=True, description="Include per-item results once the job has finished."
    ),
) -> BatchStatusResponse:
    """Check on a batch job.

    ``status`` progresses ``pending`` -> ``running`` -> ``completed`` or
    ``failed``. While running, ``progress_percent`` updates after every image.

    Results appear once the job reaches a terminal state. A job can be
    ``completed`` while ``failed_items`` is greater than zero: individual
    images may fail without failing the batch, and each carries its own error.
    """
    from celery.result import AsyncResult

    from worker.celery_app import celery_app

    result = AsyncResult(job_id, app=celery_app)

    # Celery reports PENDING for both "queued" and "never existed", so the
    # database is the authority on whether the job is real.
    from api.services.db_service import get_db_service

    db = get_db_service()
    record = await db.get_batch_job(job_id)

    # Reading `result.state` talks to the Celery result backend (Redis). If
    # that is down, the raw ConnectionError would surface as an opaque 500;
    # translate it into a 503 that tells the caller their job is not lost,
    # the status store is simply unreachable.
    try:
        state = result.state
    except Exception as exc:
        logger.error(
            "batch_status_backend_unavailable",
            extra={"job_id": job_id, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise ServiceUnavailableError(
            "Job status is temporarily unavailable because the result store "
            "cannot be reached. The job itself is unaffected; retry shortly.",
            details={"job_id": job_id},
            internal_message=f"celery result backend error: {type(exc).__name__}: {exc}",
        ) from exc

    if (record is None and state == "PENDING") or not _owns(record, principal):
        raise JobNotFoundError(
            "No batch job exists with that id. It may have expired: results "
            "are retained for 24 hours.",
            details={"job_id": job_id},
        )

    status_map = {
        "PENDING": JobStatus.PENDING,
        "STARTED": JobStatus.RUNNING,
        "PROGRESS": JobStatus.RUNNING,
        "SUCCESS": JobStatus.COMPLETED,
        "FAILURE": JobStatus.FAILED,
        "REVOKED": JobStatus.CANCELLED,
        "RETRY": JobStatus.RUNNING,
    }
    job_status = status_map.get(state, JobStatus.PENDING)

    total = record.total_items if record else 0
    # Celery forgets a result after 24 hours and then reports PENDING again.
    # The worker writes the outcome to batch_jobs, so a finished job does not
    # turn back into a queued one.
    recorded_outcome = state == "PENDING" and record is not None and record.status != "pending"
    if recorded_outcome:
        job_status = JobStatus(record.status)
    task_type = TaskType(record.task) if record else TaskType.CLASSIFICATION
    completed = failed = 0
    results: list[BatchItemResult] | None = None
    error: str | None = None
    completed_at = None
    duration = None

    if state == "PROGRESS" and isinstance(result.info, dict):
        completed = result.info.get("completed", 0)
        failed = result.info.get("failed", 0)
        total = result.info.get("total", total)

    elif state == "SUCCESS" and isinstance(result.result, dict):
        payload = result.result
        completed = payload.get("completed_items", 0)
        failed = payload.get("failed_items", 0)
        total = payload.get("total_items", total)
        duration = payload.get("duration_seconds")
        if payload.get("completed_at"):
            completed_at = datetime.fromisoformat(payload["completed_at"])
        if include_results:
            results = [BatchItemResult(**item) for item in payload.get("results", [])]

    elif recorded_outcome:
        completed = record.completed_items
        failed = record.failed_items

    elif state == "FAILURE":
        # The exception text is operator-facing detail; keep the user-facing
        # message generic and let the correlation id carry them to the logs.
        error = "The batch job failed. Quote the job id when contacting support."
        logger.error("batch_job_failed", extra={"job_id": job_id, "error": str(result.info)[:500]})

    progress = round((completed + failed) / total * 100, 2) if total else 0.0

    return BatchStatusResponse(
        job_id=job_id,
        status=job_status,
        task=task_type,
        total_items=total,
        completed_items=completed,
        failed_items=failed,
        progress_percent=min(100.0, progress),
        results=results,
        error=error,
        submitted_at=record.submitted_at if record else datetime.now(UTC),
        started_at=record.started_at if record else None,
        completed_at=completed_at or (record.completed_at if record else None),
        duration_seconds=duration,
    )


def _owns(record: BatchJob | None, principal: Principal) -> bool:
    """False only when the job is on record as belonging to someone else.

    Another user's job answers 404, not 403, so its existence is not revealed.
    With the database down there is no record to check against, and status
    reads keep working rather than failing every caller.
    """
    return record is None or record.user_id is None or record.user_id == principal.user_id


@router.delete(
    "/{job_id}",
    summary="Cancel a queued or running batch job",
    responses={404: {"model": ErrorResponse, "description": "No job with that id."}},
)
async def cancel_batch(
    principal: CurrentPrincipal,
    correlation_id: CorrelationId,
    job_id: str = Path(description="Job id to cancel."),
) -> dict[str, object]:
    """Cancel a batch job.

    A job that has not started is simply removed from the queue. A running job
    is terminated, so any images it had already finished are lost — the result
    is not partially retrievable after cancellation.
    """
    from celery.result import AsyncResult

    from api.services.db_service import get_db_service
    from worker.celery_app import celery_app

    db = get_db_service()
    record = await db.get_batch_job(job_id)
    result = AsyncResult(job_id, app=celery_app)
    # Without this, any caller could revoke any id, including other users' jobs.
    if not _owns(record, principal) or (
        record is None and db.available and result.state == "PENDING"
    ):
        raise JobNotFoundError(
            "No batch job exists with that id.",
            details={"job_id": job_id},
        )
    if result.state in ("SUCCESS", "FAILURE"):
        return {
            "job_id": job_id,
            "cancelled": False,
            "reason": f"The job has already finished with status {result.state.lower()}.",
            "correlation_id": correlation_id,
        }

    result.revoke(terminate=True, signal="SIGTERM")

    await db.update_batch_job(
        job_id, status=JobStatus.CANCELLED.value, completed_at=datetime.now(UTC)
    )

    logger.info("batch_cancelled", extra={"job_id": job_id, "user_id": principal.user_id})
    return {"job_id": job_id, "cancelled": True, "correlation_id": correlation_id}


__all__ = ["router"]
