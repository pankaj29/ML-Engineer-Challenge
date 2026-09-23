"""Shared FastAPI dependencies.

Plain English:
    A "dependency" in FastAPI is a small function that runs before a route
    handler and hands it something it needs. Putting these in one place means
    every endpoint gets identical behaviour — and there is exactly one place
    to fix if that behaviour is wrong.

The main job here is turning "an image arrived somehow" into "validated
bytes". Callers may send an image three ways, and every endpoint accepts all
three:

    1. ``multipart/form-data`` file upload — what a browser or ``curl -F`` sends.
    2. JSON with ``image_base64`` — convenient for API clients.
    3. JSON with ``image_url`` — we fetch it, behind the SSRF checks.

Whichever arrives, the route handler receives the same validated ``bytes``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, UploadFile

from api.config import Settings, settings
from api.exceptions import (
    ImageTooLargeError,
    InvalidImageError,
    ServiceUnavailableError,
    ValidationError,
)
from api.logging_config import get_logger
from api.middleware.auth import Principal, get_principal
from api.middleware.rate_limit import RateLimiter, get_rate_limiter
from api.models.schemas import ImagePayload
from api.services.cache_service import CacheService, get_cache_service
from api.services.inference_service import InferenceService, get_inference_service
from api.services.model_service import ModelService, get_model_service
from api.utils.validators import validate_image_bytes, validate_image_url

logger = get_logger(__name__)

# How long we will wait for a remote image, and how much of it we will read.
_FETCH_TIMEOUT_SECONDS = 10.0


def get_settings_dep() -> Settings:
    """Dependency returning application settings."""
    return settings


async def read_upload(
    upload: UploadFile,
    *,
    config: Settings | None = None,
    field_name: str = "file",
) -> bytes:
    """Read an uploaded file, enforcing the size limit while reading.

    The limit is checked *during* streaming rather than after. Reading a 2 GB
    upload fully into memory and only then rejecting it would let a single
    request exhaust the container's memory — the check has to happen before
    the bytes accumulate.
    """
    cfg = config or settings
    limit = cfg.max_image_bytes
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ImageTooLargeError(
                f"The upload exceeds the {limit / 1_048_576:.0f} MB limit.",
                details={"limit_bytes": limit, "field": field_name},
            )
        chunks.append(chunk)

    data = b"".join(chunks)
    if not data:
        raise InvalidImageError("The uploaded file was empty.", details={"field": field_name})
    return data


async def fetch_image_url(url: str, *, config: Settings | None = None) -> bytes:
    """Download an image from a URL, with SSRF and size protection.

    The URL is validated first (scheme, port, resolved address), then fetched
    with a timeout, a redirect ban and a streaming size cap.

    Redirects are disabled deliberately: we validated the address of the
    *original* host, and following a redirect would let that host bounce us to
    an internal address we never checked.
    """
    cfg = config or settings
    validate_image_url(url)

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ServiceUnavailableError(
            "Fetching images by URL is not available on this server.",
            internal_message="httpx is not installed",
        ) from exc

    limit = cfg.max_image_bytes
    try:
        async with (
            httpx.AsyncClient(
                timeout=_FETCH_TIMEOUT_SECONDS,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=10),
            ) as client,
            client.stream("GET", url) as response,
        ):
            if response.status_code >= 400:
                raise ValidationError(
                    f"The image URL returned HTTP {response.status_code}.",
                    details={"status_code": response.status_code},
                )

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise ImageTooLargeError(
                    f"The remote image is larger than the {limit / 1_048_576:.0f} MB limit.",
                    details={"size_bytes": int(declared), "limit_bytes": limit},
                )

            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes(64 * 1024):
                total += len(chunk)
                if total > limit:
                    # A server can lie about (or omit) Content-Length, so
                    # the real enforcement is here, while streaming.
                    raise ImageTooLargeError(
                        f"The remote image exceeds the {limit / 1_048_576:.0f} MB limit.",
                        details={"limit_bytes": limit},
                    )
                chunks.append(chunk)
    except httpx.TimeoutException as exc:
        raise ValidationError(
            "Timed out while downloading the image URL.",
            details={"timeout_seconds": _FETCH_TIMEOUT_SECONDS},
            internal_message=str(exc),
        ) from exc
    except httpx.HTTPError as exc:
        raise ValidationError(
            "The image URL could not be downloaded.",
            internal_message=f"{type(exc).__name__}: {exc}",
        ) from exc

    return b"".join(chunks)


async def resolve_image_bytes(
    payload: ImagePayload | None = None,
    upload: UploadFile | None = None,
    *,
    config: Settings | None = None,
    field_name: str = "image",
) -> bytes:
    """Get validated image bytes from whichever way the caller supplied them.

    Args:
        payload: Parsed JSON body carrying ``image_base64`` or ``image_url``.
        upload: Multipart file upload.
        field_name: Name used in error messages.

    Returns:
        Raw bytes that have passed every validation check.

    Raises:
        ValidationError: Neither or both sources were supplied.
    """
    cfg = config or settings

    if upload is not None and payload is not None:
        raise ValidationError(
            "Send either a file upload or a JSON body, not both.",
            details={"field": field_name},
        )

    if upload is not None:
        data = await read_upload(upload, config=cfg, field_name=field_name)
    elif payload is not None:
        if payload.image_url:
            data = await fetch_image_url(payload.image_url, config=cfg)
        else:
            try:
                decoded = payload.decode()
            except ValueError as exc:
                raise InvalidImageError(
                    "image_base64 is not valid base64.",
                    details={"field": field_name},
                    internal_message=str(exc),
                ) from exc
            if decoded is None:
                raise ValidationError("No image data was supplied.", details={"field": field_name})
            data = decoded
    else:
        raise ValidationError(
            "No image supplied. Send a multipart file upload, or a JSON body "
            "containing image_base64 or image_url.",
            details={"field": field_name},
        )

    validate_image_bytes(data, config=cfg, field_name=field_name)
    return data


def get_correlation_id_dep(request: Request) -> str:
    """Dependency returning this request's correlation id."""
    return getattr(request.state, "correlation_id", "")


# Annotated aliases, so route signatures stay short and readable.
CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
Models = Annotated[ModelService, Depends(get_model_service)]
Cache = Annotated[CacheService, Depends(get_cache_service)]
Inference = Annotated[InferenceService, Depends(get_inference_service)]
Limiter = Annotated[RateLimiter, Depends(get_rate_limiter)]
CorrelationId = Annotated[str, Depends(get_correlation_id_dep)]
AppSettings = Annotated[Settings, Depends(get_settings_dep)]


__all__ = [
    "AppSettings",
    "Cache",
    "CorrelationId",
    "CurrentPrincipal",
    "Inference",
    "Limiter",
    "Models",
    "fetch_image_url",
    "get_correlation_id_dep",
    "get_settings_dep",
    "read_upload",
    "resolve_image_bytes",
]
