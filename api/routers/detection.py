"""Object detection endpoint.

Plain English:
    "Here is a picture. What objects are in it, and where?" The model returns
    a list of boxes, each with a class name, a confidence and pixel
    coordinates.

Coordinates are returned in the pixel space of the image the caller
*uploaded*, not the resized image the model saw. The un-letterboxing that
makes this true happens in
:func:`~api.utils.image_processing.scale_boxes_to_original`.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from api.dependencies import CurrentPrincipal, Inference, resolve_image_bytes
from api.logging_config import get_logger
from api.middleware.monitoring import record_inference
from api.models.responses import DetectionResponse, ErrorResponse
from api.models.schemas import DetectionRequest, RuntimeFormat
from api.services.prediction_log import record_prediction

logger = get_logger(__name__)

router = APIRouter(prefix="/detect", tags=["Detection"])

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
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect objects in an image (JSON body)",
    responses=_ERROR_RESPONSES,
)
async def detect(
    body: DetectionRequest,
    principal: CurrentPrincipal,
    inference: Inference,
) -> DetectionResponse:
    """Detect objects in a single image supplied as base64 or a URL.

    Two thresholds control the output:

    * ``confidence_threshold`` — drop boxes the model is unsure about. Raise
      it to reduce false positives, lower it to catch more objects.
    * ``iou_threshold`` — how much two boxes may overlap before the lower
      scoring one is treated as a duplicate of the same object.

    Example response::

        {
          "detections": [
            {
              "class_id": 0,
              "label": "person",
              "confidence": 0.91,
              "box": {"x1": 34.0, "y1": 120.5, "x2": 210.2, "y2": 480.0}
            }
          ],
          "count": 1,
          "image_width": 640,
          "image_height": 480
        }

    Box coordinates are absolute pixels in the original uploaded image, with
    ``x1,y1`` the top-left corner and ``x2,y2`` the bottom-right.
    """
    image_bytes = await resolve_image_bytes(payload=body, field_name="image")

    started = time.perf_counter()
    response = await inference.detect(
        image_bytes,
        confidence_threshold=body.confidence_threshold,
        iou_threshold=body.iou_threshold,
        max_detections=body.max_detections,
        class_filter=body.class_filter,
        model_name=body.model_name,
        model_version=body.model_version,
        runtime=body.runtime,
        image_id=body.image_id,
    )

    record_inference(
        task="detection",
        model=response.model.name,
        version=response.model.version,
        runtime=response.model.runtime.value,
        duration_seconds=time.perf_counter() - started,
    )
    record_prediction(
        task="detection",
        response=response,
        image_bytes=image_bytes,
        principal=principal,
    )
    return response


@router.post(
    "/upload",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect objects in an uploaded image file (multipart)",
    responses=_ERROR_RESPONSES,
)
async def detect_upload(
    principal: CurrentPrincipal,
    inference: Inference,
    file: Annotated[UploadFile, File(description="Image file to run detection on.")],
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    iou_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.45,
    max_detections: Annotated[int, Form(ge=1, le=300)] = 100,
    model_name: Annotated[str | None, Form()] = None,
    model_version: Annotated[str | None, Form()] = None,
    runtime: Annotated[RuntimeFormat | None, Form()] = None,
) -> DetectionResponse:
    """Detect objects in an image sent as a multipart file upload.

    Example::

        curl -X POST http://localhost:8000/api/v1/detect/upload \\
             -H "X-API-Key: your-key" \\
             -F "file=@street.jpg" -F "confidence_threshold=0.4"
    """
    image_bytes = await resolve_image_bytes(upload=file, field_name="file")

    started = time.perf_counter()
    response = await inference.detect(
        image_bytes,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        model_name=model_name,
        model_version=model_version,
        runtime=runtime,
        image_id=file.filename,
    )

    record_inference(
        task="detection",
        model=response.model.name,
        version=response.model.version,
        runtime=response.model.runtime.value,
        duration_seconds=time.perf_counter() - started,
    )
    record_prediction(
        task="detection",
        response=response,
        image_bytes=image_bytes,
        principal=principal,
    )
    return response


__all__ = ["router"]
