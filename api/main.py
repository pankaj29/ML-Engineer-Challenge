"""FastAPI application entry point.

Plain English:
    This file assembles the service. It creates the app, decides what happens
    at startup and shutdown, stacks the middleware in the right order, and
    mounts every route.

**Middleware order matters and is not arbitrary.** Requests travel down the
stack and responses travel back up, so the outermost middleware sees the
request first and the response last:

    Request  ->  Monitoring  ->  CORS  ->  Auth  ->  RateLimit  ->  routes
    Response <-  Monitoring  <-  CORS  <-  Auth  <-  RateLimit  <-  routes

* **Monitoring is outermost** so that it assigns the correlation id before
  anything else runs, and so that it times and logs *every* request —
  including the ones rejected by authentication.
* **Auth comes before rate limiting** because the limit depends on the
  caller's tier, which is only known once they are identified.
* **Rate limiting is innermost** so that a rejected request never reaches a
  route handler and never touches a model.

Startup is fail-soft by design: if Redis or PostgreSQL is unreachable, the API
still starts and serves predictions, reporting itself *degraded*. A missing
cache should not be an outage. The one thing it will not do is start with
insecure configuration in production — that fails loudly, in
:mod:`api.config`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from api.config import settings
from api.exceptions import register_exception_handlers
from api.logging_config import configure_logging, get_logger
from api.middleware.auth import AuthMiddleware
from api.middleware.monitoring import MonitoringMiddleware, models_loaded, set_app_info
from api.middleware.rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    get_rate_limiter,
    set_rate_limiter,
)
from api.routers import (
    auth as auth_router,
    batch,
    classification,
    detection,
    health,
    metrics,
    models as models_router,
    similarity,
)
from api.services.cache_service import CacheService, set_cache_service
from api.services.db_service import DatabaseService, set_db_service
from api.services.inference_service import InferenceService, set_inference_service
from api.services.model_service import ModelService, set_model_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start up and shut down the application's shared resources.

    Everything expensive and long-lived — connection pools, loaded models — is
    created exactly once here and torn down cleanly on exit.
    """
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "application_starting",
        extra={
            "version": settings.app_version,
            "environment": settings.environment.value,
            "device": settings.resolve_device(),
            "preferred_runtime": settings.preferred_runtime,
        },
    )
    health.mark_started()
    set_app_info(settings.app_version, settings.environment.value)

    # --- Models -----------------------------------------------------------
    model_service = ModelService()
    set_model_service(model_service)

    # --- Cache, database and rate limiter, all connected concurrently ------
    cache_service = CacheService()
    db_service = DatabaseService()
    # get_rate_limiter(), NOT RateLimiter(). create_app() already built a
    # limiter and handed that exact object to RateLimitMiddleware. Building a
    # second one here and connecting *it* left the middleware holding an
    # instance on which connect() was never called - so `_available` stayed
    # False and every request silently used the per-process _LocalBucket
    # fallback. Rate limits were therefore multiplied by the replica count,
    # with nothing in the logs beyond a single degraded warning.
    rate_limiter = get_rate_limiter()
    set_cache_service(cache_service)
    set_db_service(db_service)
    set_rate_limiter(rate_limiter)

    cache_ok, db_ok, limiter_ok = await asyncio.gather(
        cache_service.connect(),
        db_service.connect(create_tables=not settings.is_production),
        rate_limiter.connect(),
        return_exceptions=False,
    )

    inference_service = InferenceService(model_service, cache_service)
    set_inference_service(inference_service)
    app.state.inference_service = inference_service

    # --- Similarity index -------------------------------------------------
    # Create the pgvector table at startup rather than on the first index
    # call, so a missing extension shows up in the startup log instead of as a
    # failed user request. ensure_schema() returns False rather than raising:
    # similarity is one feature, and it should degrade without stopping boot.
    similarity_ok = True
    if settings.similarity_backend == "pgvector":
        from api.services.similarity_index import get_similarity_index

        similarity_ok = await get_similarity_index().ensure_schema()

    # --- Warm the models --------------------------------------------------
    if settings.eager_model_load:
        report = await model_service.warmup()
        logger.info("model_warmup_complete", extra={"report": report})
    models_loaded.set(model_service.health()["loaded"])

    degraded = [
        name
        for name, ok in (
            ("cache", cache_ok),
            ("database", db_ok),
            ("rate_limiter", limiter_ok),
            ("similarity_index", similarity_ok),
        )
        if not ok
    ]
    logger.info(
        "application_started",
        extra={
            "models_loaded": model_service.health()["loaded"],
            "degraded_components": degraded or None,
        },
    )
    if degraded:
        logger.warning(
            "starting_in_degraded_mode",
            extra={
                "components": degraded,
                "impact": (
                    "cache: slower repeat requests; "
                    "database: no inference logging; "
                    "rate_limiter: per-process limits only"
                ),
            },
        )

    yield

    # --- Shutdown ---------------------------------------------------------
    logger.info("application_stopping")
    await asyncio.gather(
        cache_service.close(),
        db_service.close(),
        rate_limiter.close(),
        return_exceptions=True,
    )
    model_service.unload_all()
    logger.info("application_stopped")


def create_app(*, testing: bool = False) -> FastAPI:
    """Build the FastAPI application.

    Args:
        testing: Skip the lifespan handler so tests can inject fake services
            without connecting to real infrastructure.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production multi-model computer vision API serving image "
            "classification, object detection and image similarity search."
        ),
        lifespan=None if testing else lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        # Hide the default 422 response schema; ours is documented explicitly
        # on each route, in the standard error envelope.
        responses={},
    )

    register_exception_handlers(app)

    # --- Routes -----------------------------------------------------------
    prefix = settings.api_prefix
    app.include_router(auth_router.router, prefix=prefix)
    app.include_router(classification.router, prefix=prefix)
    app.include_router(detection.router, prefix=prefix)
    app.include_router(similarity.router, prefix=prefix)
    app.include_router(batch.router, prefix=prefix)
    app.include_router(models_router.router, prefix=prefix)
    app.include_router(health.router, prefix=prefix)
    app.include_router(metrics.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        """Service banner with links to the docs and health endpoints."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment.value,
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "health": f"{prefix}/health",
            "metrics": f"{prefix}/metrics",
            "endpoints": {
                "classify": f"{prefix}/classify",
                "detect": f"{prefix}/detect",
                "similarity": f"{prefix}/similarity/search",
                "batch": f"{prefix}/batch",
                "models": f"{prefix}/models",
            },
        }

    # --- Middleware -------------------------------------------------------
    # Added innermost-first: Starlette applies the LAST added as the
    # outermost layer, so this reversed order produces the stack documented
    # in the module docstring.
    rate_limiter = RateLimiter()
    set_rate_limiter(rate_limiter)
    app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
    app.add_middleware(MonitoringMiddleware)

    app.openapi = lambda: _custom_openapi(app)  # type: ignore[method-assign]
    return app


def _custom_openapi(app: FastAPI) -> dict[str, Any]:
    """Extend the generated OpenAPI spec with authentication and usage notes.

    FastAPI generates most of the spec from the route signatures. What it
    cannot know is how authentication works or what the rate limits are, so
    that is added here — making ``/docs`` a complete reference rather than
    just a list of endpoints.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=(
            f"{app.description}\n\n"
            "## Authentication\n\n"
            "Every endpoint except `/health`, `/metrics` and the docs requires "
            "credentials, supplied either way:\n\n"
            "- `X-API-Key: <your-key>` header, or\n"
            "- `Authorization: Bearer <jwt>` header.\n\n"
            "## Rate limits\n\n"
            "Limits are applied per API key, per minute, using a token bucket "
            "that permits a short burst:\n\n"
            f"| Tier | Requests/minute | Max batch size |\n"
            f"| --- | ---: | ---: |\n"
            f"| free | {settings.rate_limit_free_rpm} | 5 |\n"
            f"| basic | {settings.rate_limit_basic_rpm} | 16 |\n"
            f"| pro | {settings.rate_limit_pro_rpm} | 32 |\n"
            f"| enterprise | {settings.rate_limit_enterprise_rpm} | {settings.max_batch_size} |\n\n"
            "Every response carries `X-RateLimit-Limit` and "
            "`X-RateLimit-Remaining`. A rejected request returns HTTP 429 with "
            "a `Retry-After` header.\n\n"
            "## Errors\n\n"
            "Every error uses the same envelope:\n\n"
            "```json\n"
            "{\n"
            '  "error": {\n'
            '    "code": "IMAGE_TOO_LARGE",\n'
            '    "message": "The uploaded image exceeds the maximum allowed size.",\n'
            '    "details": {"size_bytes": 12582912, "limit_bytes": 10485760},\n'
            '    "correlation_id": "0f6c1d...",\n'
            '    "timestamp": "2026-09-22T10:00:00Z"\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "Branch on `code`, never on `message`. Quote `correlation_id` in "
            "support requests: every log line for your request carries it.\n\n"
            "## Images\n\n"
            f"Maximum {settings.max_image_bytes // 1_048_576} MB, formats "
            f"{', '.join(sorted(settings.allowed_formats_set))}. Send one of "
            "`image_base64`, `image_url`, or a multipart upload to the "
            "`/upload` variant of each endpoint."
        ),
        routes=app.routes,
    )

    schema["components"] = schema.get("components", {})
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key issued to your account. Determines your tier.",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Short-lived JWT access token.",
        },
    }
    schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
    schema["tags"] = [
        {"name": "Classification", "description": "Identify what an image contains."},
        {"name": "Detection", "description": "Locate objects within an image."},
        {"name": "Similarity", "description": "Embed images and search for visually similar ones."},
        {"name": "Batch", "description": "Submit many images for background processing."},
        {"name": "Models", "description": "Discover available models, versions and metrics."},
        {"name": "Health", "description": "Liveness, readiness and dependency status."},
        {"name": "Monitoring", "description": "Prometheus metrics."},
    ]

    app.openapi_schema = schema
    return schema


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # our own logging is configured in the lifespan handler
    )
