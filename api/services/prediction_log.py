"""Record predictions to the inference log.

Plain English:
    Every prediction the API makes gets a row here: which model and version,
    what it said, how confident it was, how long it took. Nothing user-facing
    depends on it. Everything that decides whether the model is still good
    does.

This existed as a table, an ORM model and a `log_inference` method, and
nothing called it. The consequence was quiet and total: drift detection reads
this table and always found it empty, so it never reported drift; the
retraining loop asked drift and was always told there was nothing to do; and a
canary could serve traffic for a week with no way to compare it against
stable. Every layer reported success.

Two properties matter here.

**It never blocks the response.** The prediction is already computed and the
user is waiting. The write is scheduled and the handler returns.

**It never raises.** A logging failure must not turn a successful prediction
into a 500. `log_inference` already swallows its errors; this adds the same
guarantee around building the record, because a missing field would otherwise
raise inside the task.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

from api.logging_config import get_logger
from api.services.db_service import get_db_service

if TYPE_CHECKING:
    from api.middleware.auth import Principal

logger = get_logger(__name__)

# Keeps a reference to in-flight tasks. Without one, asyncio only holds a weak
# reference and a write can be garbage-collected mid-flight, which loses rows
# under load and nowhere else.
_pending: set[asyncio.Task[Any]] = set()


def image_fingerprint(image_bytes: bytes) -> str:
    """SHA-256 of the image.

    The hash, never the image. It is enough to spot the same picture arriving
    twice, and it keeps user content out of a table that is queried for
    analytics and kept for a long time.
    """
    return hashlib.sha256(image_bytes).hexdigest()


def _top_result(results: list[Any]) -> tuple[str | None, float | None]:
    if not results:
        return None, None
    first = results[0]
    label = getattr(first, "label", None) or getattr(first, "class_name", None)
    confidence = getattr(first, "confidence", None) or getattr(first, "score", None)
    return (str(label) if label is not None else None, float(confidence) if confidence else None)


def record_prediction(
    *,
    task: str,
    response: Any,
    image_bytes: bytes,
    principal: Principal | None = None,
    results: list[Any] | None = None,
) -> None:
    """Schedule a row for this prediction. Returns immediately.

    Args:
        task: ``"classification"``, ``"detection"`` or ``"similarity"``.
        response: The response object, for its model info and timings.
        image_bytes: Hashed, not stored.
        principal: Who called, for per-user analysis. None when anonymous.
        results: The predictions, for the top label and count. Defaults to
            ``response.predictions`` or ``response.detections``.
    """
    try:
        db = get_db_service()
        if not db.available:
            return

        if results is None:
            results = (
                getattr(response, "predictions", None)
                or getattr(response, "detections", None)
                or getattr(response, "results", None)
                or []
            )

        label, confidence = _top_result(list(results))
        model = response.model
        timing = getattr(response, "timing", None)

        record = {
            "correlation_id": str(response.correlation_id),
            "user_id": getattr(principal, "user_id", None),
            "user_tier": getattr(getattr(principal, "tier", None), "value", None),
            "task": task,
            "model_name": model.name,
            "model_version": model.version,
            "runtime": getattr(model.runtime, "value", str(model.runtime)),
            "device": getattr(model, "device", None) or "cpu",
            "image_hash": image_fingerprint(image_bytes),
            "image_bytes": len(image_bytes),
            "top_label": label,
            "top_confidence": confidence,
            "num_results": len(list(results)),
            "preprocess_ms": getattr(timing, "preprocess_ms", None),
            "inference_ms": getattr(timing, "inference_ms", None),
            "postprocess_ms": getattr(timing, "postprocess_ms", None),
            "total_ms": getattr(timing, "total_ms", None),
            "success": True,
        }
    except Exception as exc:
        # Building the record must not break the request either.
        logger.warning(
            "prediction_log_skipped",
            extra={"error": f"{type(exc).__name__}: {exc}", "task": task},
        )
        return

    # Check for a loop before building the coroutine. Creating one and then
    # failing to schedule it leaves an un-awaited coroutine, which Python
    # warns about and which is a small leak on every call.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop, so nothing to schedule onto. Happens in synchronous callers
        # and in tests; not an error.
        return

    handle = asyncio.create_task(_write(record))
    _pending.add(handle)
    handle.add_done_callback(_pending.discard)


async def _write(record: dict[str, Any]) -> None:
    try:
        await get_db_service().log_inference(record)
    except Exception as exc:  # pragma: no cover - log_inference already guards
        logger.warning("prediction_log_failed", extra={"error": str(exc)})


__all__ = ["image_fingerprint", "record_prediction"]
