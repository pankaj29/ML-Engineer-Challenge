"""Application exception hierarchy and HTTP error handlers.

The idea in plain English:
    When something goes wrong we want two different messages. The *user* gets
    a short, friendly sentence telling them what to fix. The *operator* gets a
    full log line with a correlation ID, the stack trace and the internal
    details. This module is where those two audiences are separated.

Every error the API returns has the same JSON shape, so clients only ever need
to write one error-handling branch::

    {
      "error": {
        "code": "IMAGE_TOO_LARGE",
        "message": "The uploaded image is larger than the 10 MB limit.",
        "details": {"size_bytes": 12345678, "limit_bytes": 10485760},
        "correlation_id": "0f6c...",
        "timestamp": "2026-09-22T10:00:00Z"
      }
    }
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Base class for every error this application raises on purpose.

    Attributes:
        message: Friendly, user-facing sentence. Must never contain internal
            paths, SQL, secrets or stack details.
        code: Stable, machine-readable string (``SCREAMING_SNAKE_CASE``).
            Clients branch on this, never on the message text.
        status_code: HTTP status to return.
        details: Extra structured context that is safe to show the user.
        internal_message: Operator-only context. Logged, never returned.
    """

    code: str = "INTERNAL_ERROR"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    message: str = "An unexpected error occurred. Please try again."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        internal_message: str | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details or {}
        self.internal_message = internal_message
        super().__init__(self.message)

    def to_payload(self, correlation_id: str | None = None) -> dict[str, Any]:
        """Render the error as the standard JSON body."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        }


# --- 4xx: the caller can fix these -----------------------------------------
class ValidationError(AppError):
    """The request was structurally wrong (bad field, bad value)."""

    code = "VALIDATION_ERROR"
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    message = "The request could not be validated. Check the details field."


class InvalidImageError(AppError):
    """The uploaded bytes are not a usable image."""

    code = "INVALID_IMAGE"
    status_code = status.HTTP_400_BAD_REQUEST
    message = "The uploaded file is not a valid, readable image."


class ImageTooLargeError(AppError):
    """The upload exceeds the configured size or pixel budget."""

    code = "IMAGE_TOO_LARGE"
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    message = "The uploaded image exceeds the maximum allowed size."


class UnsupportedFormatError(AppError):
    """The image decoded, but its format is not on the allow-list."""

    code = "UNSUPPORTED_FORMAT"
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    message = "The image format is not supported."


class BatchTooLargeError(AppError):
    """Too many items in a single batch request."""

    code = "BATCH_TOO_LARGE"
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    message = "The batch contains more items than the maximum allowed."


class AuthenticationError(AppError):
    """No credentials, or credentials we cannot verify."""

    code = "AUTHENTICATION_FAILED"
    status_code = status.HTTP_401_UNAUTHORIZED
    message = "Authentication failed. Provide a valid API key or bearer token."


class AuthorizationError(AppError):
    """Valid credentials, but not allowed to do this."""

    code = "PERMISSION_DENIED"
    status_code = status.HTTP_403_FORBIDDEN
    message = "Your account tier is not permitted to perform this action."


class RateLimitError(AppError):
    """The caller exceeded their tier's request allowance."""

    code = "RATE_LIMIT_EXCEEDED"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    message = "Rate limit exceeded. Slow down and retry after the indicated delay."


class ModelNotFoundError(AppError):
    """The requested model name or version is not in the registry."""

    code = "MODEL_NOT_FOUND"
    status_code = status.HTTP_404_NOT_FOUND
    message = "The requested model or version is not available."


class JobNotFoundError(AppError):
    """The requested background job id does not exist."""

    code = "JOB_NOT_FOUND"
    status_code = status.HTTP_404_NOT_FOUND
    message = "No background job exists with that id."


# --- 5xx: we have to fix these ---------------------------------------------
class InferenceError(AppError):
    """The model was loaded but failed while predicting."""

    code = "INFERENCE_FAILED"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "The model failed to process this input. Please try again."


class ModelLoadError(AppError):
    """A model artifact could not be loaded into memory."""

    code = "MODEL_LOAD_FAILED"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "The model is not ready to serve requests yet."


class ServiceUnavailableError(AppError):
    """A dependency (database, cache, broker) is down."""

    code = "SERVICE_UNAVAILABLE"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "A required downstream service is temporarily unavailable."


class InferenceTimeoutError(AppError):
    """Inference ran past its deadline and was abandoned."""

    code = "INFERENCE_TIMEOUT"
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    message = "Inference took too long and was cancelled. Try a smaller image."


class OverloadedError(AppError):
    """The concurrency limiter rejected the request to protect the service."""

    code = "SERVICE_OVERLOADED"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "The service is at capacity. Please retry shortly."


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _correlation_id(request: Request) -> str | None:
    """Pull the correlation id that the logging middleware attached."""
    return getattr(request.state, "correlation_id", None)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Turn a deliberate :class:`AppError` into the standard JSON response."""
    cid = _correlation_id(request)
    log = logger.error if exc.status_code >= 500 else logger.warning
    log(
        "request_failed",
        extra={
            "correlation_id": cid,
            "error_code": exc.code,
            "status_code": exc.status_code,
            "path": request.url.path,
            "internal_message": exc.internal_message,
            "details": exc.details,
        },
        exc_info=exc.status_code >= 500,
    )
    headers = {"X-Correlation-ID": cid} if cid else {}
    if isinstance(exc, RateLimitError) and "retry_after_seconds" in exc.details:
        headers["Retry-After"] = str(int(exc.details["retry_after_seconds"]))
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_payload(cid),
        headers=headers,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Reshape FastAPI's own validation errors into our standard envelope.

    Without this, pydantic errors come back in a different shape from every
    other error the API produces, forcing clients to handle two formats.
    """
    cid = _correlation_id(request)
    fields = [
        {
            "field": ".".join(str(p) for p in err.get("loc", []) if p != "body"),
            "problem": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    logger.warning(
        "request_validation_failed",
        extra={"correlation_id": cid, "path": request.url.path, "fields": fields},
    )
    err = ValidationError(details={"fields": fields})
    return JSONResponse(
        status_code=err.status_code,
        content=err.to_payload(cid),
        headers={"X-Correlation-ID": cid} if cid else {},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Wrap plain ``HTTPException``s (e.g. 404 from the router) in our envelope."""
    cid = _correlation_id(request)
    code = {
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        401: "AUTHENTICATION_FAILED",
        403: "PERMISSION_DENIED",
    }.get(exc.status_code, "HTTP_ERROR")
    err = AppError(
        message=str(exc.detail),
        code=code,
        status_code=exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=err.to_payload(cid),
        headers={"X-Correlation-ID": cid} if cid else {},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence for bugs we did not anticipate.

    The user gets a generic sentence plus the correlation id (so support can
    find the incident); the full traceback goes only to the logs.
    """
    cid = _correlation_id(request)
    logger.exception(
        "unhandled_exception",
        extra={
            "correlation_id": cid,
            "path": request.url.path,
            "exception_type": type(exc).__name__,
        },
    )
    err = AppError(
        message=(
            "An unexpected internal error occurred. Quote the correlation id "
            "when contacting support."
        )
    )
    return JSONResponse(
        status_code=err.status_code,
        content=err.to_payload(cid),
        headers={"X-Correlation-ID": cid} if cid else {},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler above to the FastAPI application."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = [
    "AppError",
    "AuthenticationError",
    "AuthorizationError",
    "BatchTooLargeError",
    "ImageTooLargeError",
    "InferenceError",
    "InferenceTimeoutError",
    "InvalidImageError",
    "JobNotFoundError",
    "ModelLoadError",
    "ModelNotFoundError",
    "OverloadedError",
    "RateLimitError",
    "ServiceUnavailableError",
    "UnsupportedFormatError",
    "ValidationError",
    "register_exception_handlers",
]
