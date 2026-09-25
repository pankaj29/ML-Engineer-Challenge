"""Response schemas.

Plain English:
    These classes describe exactly what the API sends back. Declaring them
    explicitly (rather than returning loose dictionaries) buys three things:

    1. FastAPI generates accurate OpenAPI docs from them automatically.
    2. A field cannot silently disappear from the contract — a broken response
       fails our own tests, not the client's.
    3. Anything not declared here is stripped out, so an internal field can
       never leak into a public payload by accident.

Every inference response carries the same three housekeeping blocks:
``model`` (what ran), ``timing`` (how long it took), and the correlation id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from api.models.schemas import HealthStatus, JobStatus, RuntimeFormat, TaskType, UserTier


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------
class ModelInfo(BaseModel):
    """Which model produced a result. Present on every inference response."""

    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(description="Registry model name, e.g. 'resnet50-tiny-imagenet'.")
    version: str = Field(description="Registry version, e.g. '1.0.0'.")
    task: TaskType = Field(description="The task this model performs.")
    runtime: RuntimeFormat = Field(description="Which compiled format actually executed.")
    device: str = Field(description="'cpu' or 'cuda'.")


class TimingInfo(BaseModel):
    """Latency breakdown in milliseconds, for performance debugging.

    Splitting preprocessing from inference matters: if total latency rises,
    these numbers say immediately whether the model got slower or whether the
    system is spending its time decoding and resizing images.
    """

    preprocess_ms: float = Field(description="Decode, validate, resize and normalise.")
    inference_ms: float = Field(description="Time inside the model's forward pass.")
    postprocess_ms: float = Field(description="Turning raw tensors into JSON-ready results.")
    total_ms: float = Field(description="End-to-end time spent inside the handler.")


class BaseInferenceResponse(BaseModel):
    """Fields shared by every successful inference response."""

    model_config = ConfigDict(protected_namespaces=())

    model: ModelInfo
    timing: TimingInfo
    correlation_id: str = Field(description="Quote this id in any support request.")
    cached: bool = Field(
        default=False,
        description="True when this result was served from Redis instead of re-running the model.",
    )
    image_id: str | None = Field(
        default=None, description="Echo of the caller-supplied image_id, if any."
    )
    degraded: bool = Field(
        default=False,
        description=(
            "True when the primary model was unavailable and a fallback model "
            "or runtime served the request instead."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal notes, e.g. 'image was resized' or 'fell back to torch runtime'.",
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class ClassPrediction(BaseModel):
    """A single predicted class."""

    class_id: int = Field(description="Zero-based index of the class in the model's label set.")
    label: str = Field(description="Human-readable class name.")
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax probability, 0 to 1.")
    rank: int = Field(ge=1, description="1 = highest scoring.")


class ClassificationResponse(BaseInferenceResponse):
    """Response for ``POST /api/v1/classify``."""

    predictions: list[ClassPrediction] = Field(
        description="Top-k predictions, highest confidence first."
    )
    top_prediction: ClassPrediction | None = Field(
        default=None,
        description="Convenience copy of predictions[0]; null if the threshold filtered everything out.",
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    """Axis-aligned box in absolute pixel coordinates of the original image.

    Coordinates are absolute (not normalised) so that a caller can draw them
    directly onto the image they uploaded without knowing our internal resize
    factor. ``x1,y1`` is the top-left corner; ``x2,y2`` is bottom-right.
    """

    x1: float = Field(ge=0, description="Left edge, pixels.")
    y1: float = Field(ge=0, description="Top edge, pixels.")
    x2: float = Field(ge=0, description="Right edge, pixels.")
    y2: float = Field(ge=0, description="Bottom edge, pixels.")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class Detection(BaseModel):
    """One detected object."""

    class_id: int = Field(description="Zero-based class index in the detector's label set.")
    label: str = Field(description="Human-readable class name, e.g. 'person'.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence, 0 to 1.")
    box: BoundingBox


class DetectionResponse(BaseInferenceResponse):
    """Response for ``POST /api/v1/detect``."""

    detections: list[Detection] = Field(description="Objects found, highest confidence first.")
    count: int = Field(ge=0, description="Number of detections returned after filtering.")
    image_width: int = Field(description="Width of the original image in pixels.")
    image_height: int = Field(description="Height of the original image in pixels.")


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
class SimilarHit(BaseModel):
    """One neighbour returned by a similarity search."""

    id: str = Field(description="Identifier of the indexed image.")
    score: float = Field(description="Cosine similarity, -1 to 1. Higher is more similar.")
    rank: int = Field(ge=1, description="1 = most similar.")
    label: str | None = Field(default=None, description="Label stored when the image was indexed.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Metadata stored when the image was indexed."
    )


class SimilarityResponse(BaseInferenceResponse):
    """Response for ``POST /api/v1/similarity/search``."""

    results: list[SimilarHit] = Field(description="Nearest neighbours, most similar first.")
    count: int = Field(ge=0, description="Number of neighbours returned after filtering.")
    index_size: int = Field(ge=0, description="How many vectors the index currently holds.")
    embedding: list[float] | None = Field(
        default=None, description="Query embedding; present only when requested."
    )


class EmbeddingResponse(BaseInferenceResponse):
    """Response for ``POST /api/v1/similarity/embed``."""

    embedding: list[float] = Field(description="L2-normalised embedding vector.")
    dimension: int = Field(description="Length of the embedding vector.")


class IndexResponse(BaseModel):
    """Response for ``POST /api/v1/similarity/index``."""

    id: str = Field(description="Identifier assigned to the newly indexed image.")
    index_size: int = Field(description="Index size after the insert.")
    correlation_id: str


class TokenResponse(BaseModel):
    """Response for ``POST /api/v1/auth/token``.

    Shaped like an OAuth 2.0 token response (RFC 6749) so existing clients and
    HTTP libraries can consume it without special handling.
    """

    access_token: str = Field(description="Signed JWT. Send as `Authorization: Bearer <token>`.")
    token_type: str = Field(default="bearer", description="Always `bearer`.")
    expires_in: int = Field(description="Seconds until the token expires.")
    tier: UserTier = Field(description="Tier the token carries, copied from the API key.")
    scopes: list[str] = Field(
        default_factory=list,
        description="Permissions on the token. Empty means the same access as the key.",
    )


# ---------------------------------------------------------------------------
# Batch / async jobs
# ---------------------------------------------------------------------------
class BatchSubmitResponse(BaseModel):
    """HTTP 202 response for ``POST /api/v1/batch``.

    We return a job id rather than results because batch work is queued to a
    Celery worker. Poll ``GET /api/v1/batch/{job_id}`` for progress.
    """

    job_id: str = Field(description="Poll this id for status and results.")
    status: JobStatus = Field(description="Always 'pending' at submission time.")
    task: TaskType
    total_items: int = Field(description="How many images were accepted.")
    status_url: str = Field(description="Ready-made URL to poll for this job.")
    estimated_seconds: float = Field(
        description="Rough completion estimate based on measured per-image latency."
    )
    correlation_id: str
    submitted_at: datetime


class BatchItemResult(BaseModel):
    """Outcome for one image inside a batch.

    A per-item ``error`` field means one bad image does not fail the whole
    batch — the other 63 images still return their results.
    """

    index: int = Field(description="Position of this item in the submitted list.")
    image_id: str | None = None
    success: bool
    result: dict[str, Any] | None = Field(
        default=None, description="Task-specific result payload; null when success is false."
    )
    error: dict[str, Any] | None = Field(
        default=None, description="Error code and message; null when success is true."
    )
    duration_ms: float | None = None


class BatchStatusResponse(BaseModel):
    """Response for ``GET /api/v1/batch/{job_id}``."""

    job_id: str
    status: JobStatus
    task: TaskType
    total_items: int
    completed_items: int
    failed_items: int
    progress_percent: float = Field(ge=0.0, le=100.0)
    results: list[BatchItemResult] | None = Field(
        default=None, description="Populated once the job reaches a terminal state."
    )
    error: str | None = Field(default=None, description="Set when the whole job failed.")
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None


# ---------------------------------------------------------------------------
# Models / health / metrics
# ---------------------------------------------------------------------------
class ModelMetrics(BaseModel):
    """Evaluation and runtime metrics for a registered model."""

    accuracy: float | None = Field(default=None, description="Top-1 accuracy, 0 to 1.")
    top5_accuracy: float | None = Field(default=None, description="Top-5 accuracy, 0 to 1.")
    # The same two numbers on a percentage scale. Both spellings exist because
    # the training pipeline and the regression baselines record percentages,
    # and silently rescaling one into the other is how a 78.91 becomes a
    # 7891% somewhere downstream.
    top1: float | None = Field(default=None, description="Top-1 accuracy as a percentage.")
    top5: float | None = Field(default=None, description="Top-5 accuracy as a percentage.")
    map50: float | None = Field(default=None, description="Detection mAP at IoU 0.50.")
    map50_95: float | None = Field(
        default=None, description="Detection mAP averaged over IoU .50-.95."
    )
    recall_at_1: float | None = Field(default=None, description="Similarity retrieval Recall@1.")
    p50_latency_ms: float | None = Field(default=None, description="Median measured latency.")
    p95_latency_ms: float | None = Field(
        default=None, description="95th percentile measured latency."
    )
    throughput_rps: float | None = Field(default=None, description="Measured requests per second.")
    size_mb: float | None = Field(default=None, description="On-disk artifact size.")


class ModelDescriptor(BaseModel):
    """One entry in ``GET /api/v1/models``."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    task: TaskType
    runtime: RuntimeFormat
    device: str
    loaded: bool = Field(description="True when the model is resident in memory and ready.")
    is_default: bool = Field(
        description="True when this serves requests that do not pin a version."
    )
    num_classes: int | None = None
    input_shape: list[int] | None = Field(
        default=None, description="Expected input tensor shape, e.g. [1, 3, 224, 224]."
    )
    available_runtimes: list[str] = Field(
        default_factory=list,
        description=(
            "Every runtime this model can be served with, e.g. "
            "['onnx', 'onnx_int8']. Pass one as `runtime` on an inference "
            "request to select it. Without this a caller has no way to "
            "discover that a quantised variant exists."
        ),
    )
    metrics: ModelMetrics | None = None
    description: str | None = None
    limitations: list[str] = Field(
        default_factory=list,
        description="Known failure modes, copied from the model card.",
    )
    registered_at: datetime | None = None


class ModelsResponse(BaseModel):
    """Response for ``GET /api/v1/models``."""

    models: list[ModelDescriptor]
    count: int
    defaults: dict[str, str] = Field(
        description="Default 'name:version' chosen for each task when none is pinned."
    )
    correlation_id: str


class ComponentHealth(BaseModel):
    """Health of one dependency (database, cache, a model, ...)."""

    name: str
    status: HealthStatus
    latency_ms: float | None = Field(default=None, description="How long the health probe took.")
    message: str | None = Field(default=None, description="Why it is not healthy, when it is not.")


class HealthResponse(BaseModel):
    """Response for ``GET /api/v1/health``.

    The endpoint returns HTTP 200 for healthy *and* degraded, and 503 only for
    unhealthy. That distinction matters to a load balancer: a degraded
    instance can still serve traffic and should stay in the pool.
    """

    status: HealthStatus
    version: str
    environment: str
    uptime_seconds: float
    components: list[ComponentHealth]
    timestamp: datetime
    correlation_id: str | None = None


class ErrorDetail(BaseModel):
    """Inner object of the standard error envelope."""

    code: str = Field(description="Stable machine-readable code, e.g. 'IMAGE_TOO_LARGE'.")
    message: str = Field(description="Friendly, user-facing explanation.")
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    timestamp: datetime | None = None


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns."""

    error: ErrorDetail


__all__ = [
    "BaseInferenceResponse",
    "BatchItemResult",
    "BatchStatusResponse",
    "BatchSubmitResponse",
    "BoundingBox",
    "ClassPrediction",
    "ClassificationResponse",
    "ComponentHealth",
    "Detection",
    "DetectionResponse",
    "EmbeddingResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "IndexResponse",
    "ModelDescriptor",
    "ModelInfo",
    "ModelMetrics",
    "ModelsResponse",
    "SimilarHit",
    "SimilarityResponse",
    "TimingInfo",
    "TokenResponse",
]
