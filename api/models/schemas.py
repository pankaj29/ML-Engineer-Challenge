"""Request schemas and shared enums.

Plain English:
    These classes describe exactly what a caller is allowed to send us.
    FastAPI validates every incoming request against them automatically and
    rejects anything that does not fit, before a single line of our own code
    runs. They also generate the OpenAPI documentation.

Response shapes live in :mod:`api.models.responses`.
"""

from __future__ import annotations

import base64
import binascii
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.config import settings


# ---------------------------------------------------------------------------
# Enums (shared vocabulary between requests, responses and the database)
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    """The three computer-vision tasks this system serves."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SIMILARITY = "similarity"


class UserTier(str, Enum):
    """Subscription tiers, which drive rate limits and batch size caps."""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class RuntimeFormat(str, Enum):
    """Which compiled form of a model is executing."""

    TORCH = "torch"
    TORCH_INT8 = "torch_int8"
    ONNX = "onnx"
    ONNX_INT8 = "onnx_int8"
    TENSORRT = "tensorrt"


class JobStatus(str, Enum):
    """Lifecycle of an asynchronous batch job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    """Overall service health, following the usual traffic-light convention."""

    HEALTHY = "healthy"  # everything works
    DEGRADED = "degraded"  # serving traffic, but something is wrong
    UNHEALTHY = "unhealthy"  # cannot serve traffic


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ImagePayload(BaseModel):
    """One image supplied inside a JSON body.

    Exactly one of ``image_base64`` or ``image_url`` must be provided. We
    accept base64 for JSON clients; the multipart file-upload path is offered
    separately on each endpoint for browsers and ``curl -F``.
    """

    model_config = ConfigDict(extra="forbid")

    image_base64: str | None = Field(
        default=None,
        description="Raw image bytes, base64-encoded. A `data:` URI prefix is accepted and stripped.",
    )
    image_url: str | None = Field(
        default=None,
        description="Publicly reachable HTTPS URL to fetch the image from.",
        max_length=2048,
    )
    image_id: str | None = Field(
        default=None,
        max_length=128,
        description="Caller-supplied identifier echoed back in the response, useful for batches.",
    )

    @field_validator("image_base64")
    @classmethod
    def _strip_data_uri(cls, v: str | None) -> str | None:
        """Accept both `data:image/png;base64,AAAA` and bare `AAAA`."""
        if v is None:
            return None
        if v.startswith("data:"):
            _, _, v = v.partition(",")
        v = v.strip()
        if not v:
            raise ValueError("image_base64 is empty")
        return v

    @field_validator("image_url")
    @classmethod
    def _require_https(cls, v: str | None) -> str | None:
        """Only allow http(s) URLs.

        SECURITY: schemes such as ``file://`` or ``gopher://`` would turn this
        field into a server-side request forgery (SSRF) primitive, letting a
        caller read files off the server. The fetcher in
        :mod:`api.utils.validators` applies the remaining SSRF checks.
        """
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError("image_url must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ImagePayload:
        provided = [f for f in (self.image_base64, self.image_url) if f]
        if len(provided) != 1:
            raise ValueError("provide exactly one of image_base64 or image_url")
        return self

    def decode(self) -> bytes | None:
        """Decode ``image_base64`` to raw bytes, or return None for URL inputs."""
        if not self.image_base64:
            return None
        try:
            return base64.b64decode(self.image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image_base64 is not valid base64: {exc}") from exc


class ModelSelector(BaseModel):
    """Optional model name/version pinning, shared by every inference request.

    Leaving both fields unset means "use the current default model, latest
    version", which is what almost every caller wants. Pinning is there so a
    client can hold a version steady while a new one rolls out.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Registry model name. Defaults to the task's default model.",
    )
    model_version: str | None = Field(
        default=None,
        max_length=32,
        pattern=r"^[a-zA-Z0-9_.\-]+$",
        description="Registry version, or 'latest'. Defaults to 'latest'.",
    )
    runtime: RuntimeFormat | None = Field(
        default=None,
        description="Force a specific runtime format. Defaults to the server's preferred runtime.",
    )


# ---------------------------------------------------------------------------
# Endpoint request bodies
# ---------------------------------------------------------------------------
class ClassificationRequest(ImagePayload, ModelSelector):
    """Body for ``POST /api/v1/classify``."""

    top_k: Annotated[int, Field(ge=1, le=100)] = Field(
        default=5, description="How many of the highest-scoring classes to return."
    )
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0, description="Drop predictions scoring below this probability."
    )
    include_probabilities: bool = Field(
        default=True, description="Include the full softmax probability for each class."
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "top_k": 5,
                    "confidence_threshold": 0.1,
                }
            ]
        },
    )


class DetectionRequest(ImagePayload, ModelSelector):
    """Body for ``POST /api/v1/detect``."""

    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.25, description="Minimum objectness score for a box to be reported."
    )
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.45,
        description=(
            "Non-maximum-suppression overlap threshold. Boxes overlapping more "
            "than this with a higher-scoring box are removed as duplicates."
        ),
    )
    max_detections: Annotated[int, Field(ge=1, le=300)] = Field(
        default=100, description="Hard cap on boxes returned, highest score first."
    )
    class_filter: list[str] | None = Field(
        default=None,
        max_length=80,
        description="Only return objects whose class name is in this list.",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class SimilarityRequest(ImagePayload, ModelSelector):
    """Body for ``POST /api/v1/similarity/search``."""

    top_k: Annotated[int, Field(ge=1, le=100)] = Field(
        default=10, description="How many nearest neighbours to return."
    )
    min_similarity: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        default=0.0, description="Drop neighbours scoring below this cosine similarity."
    )
    include_embedding: bool = Field(
        default=False,
        description="Also return the query image's raw embedding vector (large).",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class EmbeddingRequest(ImagePayload, ModelSelector):
    """Body for ``POST /api/v1/similarity/embed`` (vector only, no search)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class BatchItem(ImagePayload):
    """One entry inside a batch request."""

    model_config = ConfigDict(extra="forbid")


class BatchRequest(ModelSelector):
    """Body for ``POST /api/v1/batch``.

    Batches are processed by a Celery worker in the background. The endpoint
    returns a job id immediately (HTTP 202) rather than blocking, because a
    64-image batch can take far longer than a sensible HTTP timeout.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    task: TaskType = Field(description="Which model task to run over every item.")
    items: list[BatchItem] = Field(
        min_length=1,
        description="Images to process. The per-request cap depends on your tier.",
    )
    # Task-specific knobs, kept flat so one batch endpoint serves all tasks.
    top_k: Annotated[int, Field(ge=1, le=100)] = 5
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    max_detections: Annotated[int, Field(ge=1, le=300)] = 100
    callback_url: str | None = Field(
        default=None,
        max_length=2048,
        description="Optional HTTPS URL to POST the finished result to.",
    )
    priority: Annotated[int, Field(ge=0, le=9)] = Field(
        default=5, description="Queue priority; lower numbers run first."
    )

    @field_validator("items")
    @classmethod
    def _enforce_absolute_cap(cls, v: list[BatchItem]) -> list[BatchItem]:
        """Reject absurd batches early.

        This is the absolute server-wide ceiling. A second, tier-specific
        check happens in the router, where the caller's tier is known.
        """
        if len(v) > settings.max_batch_size:
            raise ValueError(
                f"batch contains {len(v)} items, the maximum is {settings.max_batch_size}"
            )
        return v

    @field_validator("callback_url")
    @classmethod
    def _callback_must_be_https(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith("https://"):
            raise ValueError("callback_url must use https://")
        return v


class IndexImageRequest(ImagePayload, ModelSelector):
    """Body for ``POST /api/v1/similarity/index`` — add an image to the index."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    label: str | None = Field(
        default=None,
        max_length=128,
        description="Human-readable label stored alongside the vector.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Arbitrary JSON metadata returned with search hits."
    )


__all__ = [
    "BatchItem",
    "BatchRequest",
    "ClassificationRequest",
    "DetectionRequest",
    "EmbeddingRequest",
    "HealthStatus",
    "ImagePayload",
    "IndexImageRequest",
    "JobStatus",
    "ModelSelector",
    "RuntimeFormat",
    "SimilarityRequest",
    "TaskType",
    "UserTier",
]
