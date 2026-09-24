"""Inference orchestration.

Plain English:
    The routers know about HTTP. The model service knows about loading files.
    This service is the bit in the middle that actually *runs a prediction*:
    check the cache, preprocess, run the model, turn raw numbers into labelled
    results, record metrics, cache the answer.

Three production concerns are handled here rather than in the routers, so that
every endpoint gets them for free:

1. **Concurrency limiting.** Inference is CPU-heavy. If 500 requests arrive at
   once and each spawns a thread, the box thrashes and everything times out.
   A semaphore caps how many run at a time; the rest wait briefly, and if the
   queue is hopeless they are rejected fast with a 503. Failing fast is kinder
   than timing out slowly.

2. **Timeouts.** A pathological input can make a model run for minutes. Every
   inference has a deadline; past it, the request is abandoned so it cannot
   hold a worker hostage.

3. **Graceful degradation.** If the pinned model will not load, we try the
   task's default model instead, and mark the response ``degraded: true``. A
   slightly worse answer beats no answer — and the flag means clients can tell
   the difference rather than silently trusting a fallback.

The heavy lifting runs in a thread pool (``asyncio.to_thread``) because ONNX
Runtime and PyTorch release the GIL during compute. That keeps the event loop
free to accept new connections while predictions are in flight.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from api.config import Settings, settings
from api.exceptions import (
    InferenceError,
    InferenceTimeoutError,
    ModelLoadError,
    ModelNotFoundError,
    OverloadedError,
)
from api.logging_config import get_correlation_id, get_logger
from api.models.responses import (
    BoundingBox,
    ClassificationResponse,
    ClassPrediction,
    Detection,
    DetectionResponse,
    EmbeddingResponse,
    ModelInfo,
    TimingInfo,
)
from api.models.schemas import RuntimeFormat, TaskType
from api.services.cache_service import CacheService, build_cache_key
from api.services.model_service import LoadedModel, ModelService
from api.utils.image_processing import (
    PreprocessResult,
    image_hash,
    preprocess,
    scale_boxes_to_original,
)

logger = get_logger(__name__)


@dataclass
class InferenceOutcome:
    """Internal result of running one model, before HTTP shaping."""

    outputs: list[np.ndarray]
    model: LoadedModel
    preprocess_result: PreprocessResult
    preprocess_ms: float
    inference_ms: float
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert raw model scores into probabilities that sum to 1.

    The ``- max`` term is not cosmetic: ``exp(1000)`` overflows to infinity in
    float32, producing NaNs. Subtracting the maximum first leaves the result
    mathematically identical while keeping every exponent at or below zero.
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-maximum suppression: remove duplicate boxes for the same object.

    A detector typically fires several times on one object. NMS keeps the
    highest-scoring box and discards any other box overlapping it by more than
    ``iou_threshold``, where IoU ("intersection over union") is the shared area
    divided by the combined area — 1.0 means identical, 0.0 means disjoint.

    Args:
        boxes: ``(N, 4)`` array of ``x1, y1, x2, y2``.
        scores: ``(N,)`` confidence per box.
        iou_threshold: Overlap above which a box is treated as a duplicate.

    Returns:
        Indices of the boxes to keep, highest score first.
    """
    if boxes.size == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[best], x1[rest])
        yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest])
        yy2 = np.minimum(y2[best], y2[rest])

        overlap = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[best] + areas[rest] - overlap
        # Guard against a zero-area box producing a division by zero.
        iou = np.where(union > 0, overlap / np.maximum(union, 1e-9), 0.0)

        order = rest[iou <= iou_threshold]

    return keep


def l2_normalize(vec: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scale a vector to unit length.

    Once vectors are unit length, their dot product *is* the cosine
    similarity, which turns an expensive similarity search into a single
    matrix multiplication.
    """
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    return vec / np.maximum(norm, 1e-12)


def _set_inflight(count: int) -> None:
    """Publish the current in-flight inference count (best-effort)."""
    try:
        from api.middleware.monitoring import inference_in_progress

        inference_in_progress.set(count)
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
class InferenceService:
    """Runs predictions with caching, timeouts, limits and fallbacks."""

    def __init__(
        self,
        model_service: ModelService,
        cache_service: CacheService,
        config: Settings | None = None,
    ) -> None:
        self.models = model_service
        self.cache = cache_service
        self.settings = config or settings
        # Caps how many inferences run at once. Sized to the box, not to the
        # traffic: exceeding it makes everything slower, not faster.
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_inferences)
        self._inflight = 0

    @property
    def inflight(self) -> int:
        """How many inferences are executing right now."""
        return self._inflight

    # ------------------------------------------------------------- internals --
    async def _acquire(self) -> None:
        """Take a concurrency slot, or fail fast when the queue is saturated.

        Waiting forever behind a full queue turns a capacity problem into a
        timeout problem. A prompt 503 lets the caller retry or shed load.
        """
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=2.0)
        except TimeoutError as exc:
            logger.warning("inference_rejected_overloaded", extra={"inflight": self._inflight})
            raise OverloadedError(
                details={
                    "inflight": self._inflight,
                    "limit": self.settings.max_concurrent_inferences,
                    "retry_after_seconds": 1,
                }
            ) from exc

    async def _resolve_model(
        self,
        task: TaskType,
        name: str | None,
        version: str | None,
        runtime: RuntimeFormat | None,
    ) -> tuple[LoadedModel, bool, list[str]]:
        """Load the requested model, falling back to the task default.

        Returns:
            ``(model, degraded, warnings)``. ``degraded`` is True when the
            request is being served by something other than what was asked for.

        Raises:
            ModelNotFoundError: The pinned model does not exist. This is
                deliberately *not* subject to fallback — a caller who typos
                a model name must get a 404 telling them so, rather than
                silently receiving predictions from a different model.
        """
        try:
            return await self.models.get(task, name, version, runtime), False, []
        except ModelLoadError as primary:
            # Nothing to fall back to if no model was pinned in the first place.
            if name is None and version is None and runtime is None:
                raise

            logger.warning(
                "model_fallback_attempt",
                extra={
                    "task": task.value,
                    "requested": f"{name}:{version}",
                    "reason": primary.internal_message or primary.message,
                },
            )
            try:
                fallback = await self.models.get(task)
            except (ModelLoadError, ModelNotFoundError):
                raise primary from None
            if fallback.entry.name == name and fallback.entry.version == version:
                # The default *is* the model that just failed; re-raise rather
                # than reporting a fallback that did not actually happen.
                raise primary from None

            warning = (
                f"requested model '{name or 'default'}:{version or 'latest'}' was "
                f"unavailable; served by '{fallback.entry.key}' instead"
            )
            logger.warning("model_fallback_used", extra={"served_by": fallback.entry.key})
            return fallback, True, [warning]

    async def _run_model(
        self,
        model: LoadedModel,
        array: np.ndarray,
    ) -> tuple[list[np.ndarray], float]:
        """Execute one forward pass under the concurrency limit and deadline."""
        await self._acquire()
        self._inflight += 1
        _set_inflight(self._inflight)
        started = time.perf_counter()
        try:
            outputs = await asyncio.wait_for(
                asyncio.to_thread(model.runtime.infer, array),
                timeout=self.settings.inference_timeout_seconds,
            )
        except TimeoutError as exc:
            logger.error(
                "inference_timeout",
                extra={
                    "model": model.entry.key,
                    "timeout_s": self.settings.inference_timeout_seconds,
                },
            )
            raise InferenceTimeoutError(
                details={
                    "model": model.entry.key,
                    "timeout_seconds": self.settings.inference_timeout_seconds,
                }
            ) from exc
        except Exception as exc:
            logger.exception("inference_failed", extra={"model": model.entry.key})
            raise InferenceError(
                details={"model": model.entry.key},
                internal_message=f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            self._inflight -= 1
            _set_inflight(self._inflight)
            self._semaphore.release()

        return outputs, (time.perf_counter() - started) * 1000.0

    async def _prepare(
        self,
        image_bytes: bytes,
        task: TaskType,
        name: str | None,
        version: str | None,
        runtime: RuntimeFormat | None,
    ) -> InferenceOutcome:
        """Resolve the model, preprocess the image and run the forward pass."""
        model, degraded, warnings = await self._resolve_model(task, name, version, runtime)

        pre_started = time.perf_counter()
        pre = await asyncio.to_thread(preprocess, image_bytes, model.preprocess_config)
        preprocess_ms = (time.perf_counter() - pre_started) * 1000.0
        warnings.extend(pre.warnings)

        outputs, inference_ms = await self._run_model(model, pre.array)

        return InferenceOutcome(
            outputs=outputs,
            model=model,
            preprocess_result=pre,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            degraded=degraded,
            warnings=warnings,
        )

    @staticmethod
    def _model_info(model: LoadedModel) -> ModelInfo:
        return ModelInfo(
            name=model.entry.name,
            version=model.entry.version,
            task=model.entry.task,
            runtime=model.runtime.format,
            device=model.runtime.device,
        )

    # --------------------------------------------------------- classification --
    async def classify(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        confidence_threshold: float = 0.0,
        include_probabilities: bool = True,
        model_name: str | None = None,
        model_version: str | None = None,
        runtime: RuntimeFormat | None = None,
        image_id: str | None = None,
        use_cache: bool = True,
    ) -> ClassificationResponse:
        """Classify one image and return the top-k predicted classes."""
        total_started = time.perf_counter()
        cid = get_correlation_id()

        params = {
            "top_k": top_k,
            "threshold": confidence_threshold,
            "probs": include_probabilities,
        }
        digest = image_hash(image_bytes)

        # --- cache lookup -------------------------------------------------
        cache_key: str | None = None
        if use_cache:
            try:
                entry = self.models.resolve(TaskType.CLASSIFICATION, model_name, model_version)
                cache_key = build_cache_key(
                    "classify",
                    digest,
                    # The fingerprint, not just name:version. Replacing weights
                    # without bumping the version would otherwise leave the
                    # cache serving the previous model's answers.
                    f"{entry.key}@{self.models.artifact_fingerprint(entry)}",
                    (runtime or RuntimeFormat(self.settings.preferred_runtime)).value,
                    params,
                )
                cached = await self.cache.get(cache_key)
            except ModelNotFoundError:
                cached = None
            if cached:
                cached["cached"] = True
                cached["correlation_id"] = cid
                cached["image_id"] = image_id
                return ClassificationResponse.model_validate(cached)

        # --- inference ----------------------------------------------------
        outcome = await self._prepare(
            image_bytes, TaskType.CLASSIFICATION, model_name, model_version, runtime
        )

        post_started = time.perf_counter()
        logits = outcome.outputs[0]
        if logits.ndim == 1:
            logits = logits[None, :]
        probs = softmax(logits.astype(np.float32))[0]

        # argpartition finds the k largest without sorting all 1000+ classes.
        k = min(top_k, probs.shape[0])
        top_idx = np.argpartition(probs, -k)[-k:]
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]

        predictions = [
            ClassPrediction(
                class_id=int(idx),
                label=outcome.model.label_for(int(idx)),
                confidence=float(probs[idx]) if include_probabilities else 0.0,
                rank=rank,
            )
            for rank, idx in enumerate(top_idx, start=1)
            if float(probs[idx]) >= confidence_threshold
        ]
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0

        response = ClassificationResponse(
            predictions=predictions,
            top_prediction=predictions[0] if predictions else None,
            model=self._model_info(outcome.model),
            timing=TimingInfo(
                preprocess_ms=round(outcome.preprocess_ms, 2),
                inference_ms=round(outcome.inference_ms, 2),
                postprocess_ms=round(postprocess_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            correlation_id=cid,
            cached=False,
            image_id=image_id,
            degraded=outcome.degraded,
            warnings=outcome.warnings,
        )

        if use_cache and cache_key and not outcome.degraded:
            # Degraded results are never cached: the fallback is temporary and
            # we do not want it served after the primary model recovers.
            await self.cache.set(cache_key, response.model_dump(mode="json"))

        return response

    # ------------------------------------------------------------- detection --
    async def detect(
        self,
        image_bytes: bytes,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        class_filter: list[str] | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        runtime: RuntimeFormat | None = None,
        image_id: str | None = None,
        use_cache: bool = True,
    ) -> DetectionResponse:
        """Detect objects in one image and return their boxes."""
        total_started = time.perf_counter()
        cid = get_correlation_id()

        params = {
            "conf": confidence_threshold,
            "iou": iou_threshold,
            "max": max_detections,
            "classes": sorted(class_filter) if class_filter else None,
        }
        digest = image_hash(image_bytes)

        cache_key: str | None = None
        if use_cache:
            try:
                entry = self.models.resolve(TaskType.DETECTION, model_name, model_version)
                cache_key = build_cache_key(
                    "detect",
                    digest,
                    # The fingerprint, not just name:version. Replacing weights
                    # without bumping the version would otherwise leave the
                    # cache serving the previous model's answers.
                    f"{entry.key}@{self.models.artifact_fingerprint(entry)}",
                    (runtime or RuntimeFormat(self.settings.preferred_runtime)).value,
                    params,
                )
                cached = await self.cache.get(cache_key)
            except ModelNotFoundError:
                cached = None
            if cached:
                cached["cached"] = True
                cached["correlation_id"] = cid
                cached["image_id"] = image_id
                return DetectionResponse.model_validate(cached)

        outcome = await self._prepare(
            image_bytes, TaskType.DETECTION, model_name, model_version, runtime
        )

        post_started = time.perf_counter()
        detections = self._decode_detections(
            outcome,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_filter=class_filter,
        )
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0

        width, height = outcome.preprocess_result.original_size
        response = DetectionResponse(
            detections=detections,
            count=len(detections),
            image_width=width,
            image_height=height,
            model=self._model_info(outcome.model),
            timing=TimingInfo(
                preprocess_ms=round(outcome.preprocess_ms, 2),
                inference_ms=round(outcome.inference_ms, 2),
                postprocess_ms=round(postprocess_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            correlation_id=cid,
            cached=False,
            image_id=image_id,
            degraded=outcome.degraded,
            warnings=outcome.warnings,
        )

        if use_cache and cache_key and not outcome.degraded:
            await self.cache.set(cache_key, response.model_dump(mode="json"))

        return response

    def _decode_detections(
        self,
        outcome: InferenceOutcome,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        class_filter: list[str] | None,
    ) -> list[Detection]:
        """Turn a YOLO output tensor into labelled boxes in original coordinates.

        YOLOv8 emits one tensor shaped ``(1, 4 + num_classes, num_anchors)``:
        the first four rows are box centre-x, centre-y, width and height; the
        remaining rows are the per-class scores for that anchor. There is no
        separate objectness score in v8 — the class score *is* the confidence.
        """
        raw = outcome.outputs[0]
        if raw.ndim == 3:
            raw = raw[0]

        # Orient the array as (attributes, anchors) regardless of which way the
        # exporter laid it out.
        attrs = raw if raw.shape[0] < raw.shape[1] else raw.T
        num_attrs = attrs.shape[0]

        if num_attrs < 5:
            logger.error("detection_output_unrecognised", extra={"shape": list(raw.shape)})
            return []

        boxes_xywh = attrs[:4, :].T  # (anchors, 4)
        class_scores = attrs[4:, :].T  # (anchors, num_classes)

        confidences = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        # A detector's class score is a probability and must lie in [0, 1].
        # Values outside that range mean the exported graph is missing its
        # final sigmoid — a real export bug worth surfacing. We clamp so one
        # odd value cannot turn into a 500, but log it so the cause is visible
        # rather than silently masked.
        if confidences.size and (confidences.min() < 0.0 or confidences.max() > 1.0):
            logger.warning(
                "detection_scores_out_of_range",
                extra={
                    "model": outcome.model.entry.key,
                    "observed_min": float(confidences.min()),
                    "observed_max": float(confidences.max()),
                    "likely_cause": "model exported without its final sigmoid activation",
                },
            )
            confidences = np.clip(confidences, 0.0, 1.0)

        keep_mask = confidences >= confidence_threshold
        if not keep_mask.any():
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        confidences = confidences[keep_mask]
        class_ids = class_ids[keep_mask]

        # centre-x/centre-y/width/height -> x1/y1/x2/y2
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:4]
        boxes_xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

        keep = nms(boxes_xyxy, confidences, iou_threshold)[:max_detections]
        if not keep:
            return []

        boxes_xyxy = scale_boxes_to_original(boxes_xyxy[keep], outcome.preprocess_result)
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        results: list[Detection] = []
        for box, conf, cid_ in zip(boxes_xyxy, confidences, class_ids, strict=False):
            label = outcome.model.label_for(int(cid_))
            if class_filter and label not in class_filter:
                continue
            results.append(
                Detection(
                    class_id=int(cid_),
                    label=label,
                    confidence=float(conf),
                    box=BoundingBox(
                        x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])
                    ),
                )
            )
        return results

    # ------------------------------------------------------------ similarity --
    async def embed(
        self,
        image_bytes: bytes,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
        runtime: RuntimeFormat | None = None,
        image_id: str | None = None,
    ) -> EmbeddingResponse:
        """Turn one image into a normalised embedding vector.

        An embedding is a list of numbers positioned so that visually similar
        images land close together. Normalising to unit length means a plain
        dot product gives the cosine similarity directly.
        """
        total_started = time.perf_counter()
        cid = get_correlation_id()

        outcome = await self._prepare(
            image_bytes, TaskType.SIMILARITY, model_name, model_version, runtime
        )

        post_started = time.perf_counter()
        raw = outcome.outputs[0]
        if raw.ndim > 2:
            # Collapse any leftover spatial dimensions, e.g. (1, 512, 1, 1).
            raw = raw.reshape(raw.shape[0], -1)
        vector = l2_normalize(raw.astype(np.float32))[0]
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0

        return EmbeddingResponse(
            embedding=[float(x) for x in vector],
            dimension=int(vector.shape[0]),
            model=self._model_info(outcome.model),
            timing=TimingInfo(
                preprocess_ms=round(outcome.preprocess_ms, 2),
                inference_ms=round(outcome.inference_ms, 2),
                postprocess_ms=round(postprocess_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            correlation_id=cid,
            image_id=image_id,
            degraded=outcome.degraded,
            warnings=outcome.warnings,
        )

    async def embed_array(
        self,
        image_bytes: bytes,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> np.ndarray:
        """Embedding as a raw NumPy vector, for the similarity index."""
        outcome = await self._prepare(
            image_bytes, TaskType.SIMILARITY, model_name, model_version, None
        )
        raw = outcome.outputs[0]
        if raw.ndim > 2:
            raw = raw.reshape(raw.shape[0], -1)
        return l2_normalize(raw.astype(np.float32))[0]


# Process-wide singleton, created in the application lifespan handler.
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """FastAPI dependency returning the shared :class:`InferenceService`."""
    global _inference_service
    if _inference_service is None:
        from api.services.cache_service import get_cache_service
        from api.services.model_service import get_model_service

        _inference_service = InferenceService(get_model_service(), get_cache_service())
    return _inference_service


def set_inference_service(service: InferenceService | None) -> None:
    """Replace the singleton. Used by the lifespan handler and by tests."""
    global _inference_service
    _inference_service = service


__all__ = [
    "InferenceOutcome",
    "InferenceService",
    "get_inference_service",
    "l2_normalize",
    "nms",
    "set_inference_service",
    "softmax",
]
