"""Image classification endpoint.

Plain English:
    "Here is a picture. What is in it?" The model returns a ranked list of
    class names with a confidence for each.

The route itself is deliberately thin. It parses the request, hands the work
to :class:`~api.services.inference_service.InferenceService`, records a metric
and returns. All the interesting behaviour — caching, fallbacks, timeouts —
lives in the service, so it is shared with the batch worker rather than
duplicated.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile, status

from api.dependencies import CurrentPrincipal, Inference, resolve_image_bytes
from api.logging_config import get_logger
from api.middleware.monitoring import record_inference
from api.models.responses import ClassificationResponse, ErrorResponse
from api.models.schemas import ClassificationRequest, RuntimeFormat

logger = get_logger(__name__)

router = APIRouter(prefix="/classify", tags=["Classification"])

# Documented error responses, so the OpenAPI spec shows what can go wrong.
_ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "The image could not be decoded."},
    401: {"model": ErrorResponse, "description": "Missing or invalid credentials."},
    413: {"model": ErrorResponse, "description": "The image exceeds the size limit."},
    415: {"model": ErrorResponse, "description": "Unsupported image format."},
    422: {"model": ErrorResponse, "description": "The request body failed validation."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded for your tier."},
    503: {"model": ErrorResponse, "description": "The model is not ready to serve."},
    504: {"model": ErrorResponse, "description": "Inference exceeded its deadline."},
}


@router.post(
    "",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify an image (JSON body)",
    responses=_ERROR_RESPONSES,
)
async def classify(
    request: Request,
    body: ClassificationRequest,
    principal: CurrentPrincipal,
    inference: Inference,
) -> ClassificationResponse:
    """Classify a single image supplied as base64 or a URL.

    Send **either** ``image_base64`` **or** ``image_url``, never both.

    Example request::

        {
          "image_base64": "iVBORw0KGgoAAA...",
          "top_k": 5,
          "confidence_threshold": 0.1
        }

    Example response::

        {
          "predictions": [
            {"class_id": 207, "label": "golden retriever", "confidence": 0.93, "rank": 1}
          ],
          "model": {"name": "resnet50", "version": "1.0.0", "runtime": "onnx", "device": "cpu"},
          "timing": {"preprocess_ms": 8.1, "inference_ms": 31.4, "total_ms": 41.9},
          "correlation_id": "0f6c...",
          "cached": false
        }
    """
    image_bytes = await resolve_image_bytes(payload=body, field_name="image")

    started = time.perf_counter()
    response = await inference.classify(
        image_bytes,
        top_k=body.top_k,
        confidence_threshold=body.confidence_threshold,
        include_probabilities=body.include_probabilities,
        model_name=body.model_name,
        model_version=body.model_version,
        runtime=body.runtime,
        image_id=body.image_id,
    )

    record_inference(
        task="classification",
        model=response.model.name,
        version=response.model.version,
        runtime=response.model.runtime.value,
        duration_seconds=time.perf_counter() - started,
    )
    return response


@router.post(
    "/upload",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify an uploaded image file (multipart)",
    responses=_ERROR_RESPONSES,
)
async def classify_upload(
    request: Request,
    principal: CurrentPrincipal,
    inference: Inference,
    file: Annotated[UploadFile, File(description="Image file to classify.")],
    top_k: Annotated[int, Form(ge=1, le=100)] = 5,
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    model_name: Annotated[str | None, Form()] = None,
    model_version: Annotated[str | None, Form()] = None,
    runtime: Annotated[RuntimeFormat | None, Form()] = None,
) -> ClassificationResponse:
    """Classify an image sent as a multipart file upload.

    This is the endpoint to use from a browser form or with::

        curl -X POST http://localhost:8000/api/v1/classify/upload \\
             -H "X-API-Key: your-key" \\
             -F "file=@cat.jpg" -F "top_k=3"
    """
    image_bytes = await resolve_image_bytes(upload=file, field_name="file")

    started = time.perf_counter()
    response = await inference.classify(
        image_bytes,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
        model_name=model_name,
        model_version=model_version,
        runtime=runtime,
        image_id=file.filename,
    )

    record_inference(
        task="classification",
        model=response.model.name,
        version=response.model.version,
        runtime=response.model.runtime.value,
        duration_seconds=time.perf_counter() - started,
    )
    return response


__all__ = ["router"]
