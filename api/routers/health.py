"""Health check endpoints.

Plain English:
    Health checks answer "should traffic be sent to this container?" Three
    different questions hide inside that, and conflating them causes outages:

    * **Liveness** (``/health/live``) — is the process alive? If this fails,
      the orchestrator should *restart* the container. It must never depend on
      the database, or a brief database blip would restart every container at
      once and turn a small problem into a total outage.

    * **Readiness** (``/health/ready``) — can this container serve a real
      request *right now*? If this fails, the orchestrator should stop routing
      traffic here but leave the container running. This one *does* check
      dependencies.

    * **Detailed health** (``/health``) — the full picture for humans and
      dashboards.

Status codes follow the same logic: ``healthy`` and ``degraded`` both return
HTTP 200, because a degraded instance can still serve traffic and should stay
in the load-balancer pool. Only ``unhealthy`` returns 503.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Response, status

from api.config import settings
from api.dependencies import Cache, Inference, Models
from api.logging_config import get_logger
from api.models.responses import ComponentHealth, HealthResponse
from api.models.schemas import HealthStatus, TaskType

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])

# Set when the application starts, so uptime is measurable.
_STARTED_AT = time.time()


def mark_started() -> None:
    """Record process start time. Called from the application lifespan."""
    global _STARTED_AT
    _STARTED_AT = time.time()


def uptime_seconds() -> float:
    """Seconds since the application finished starting up."""
    return time.time() - _STARTED_AT


async def _check_cache(cache: Cache) -> ComponentHealth:
    """Probe Redis."""
    started = time.perf_counter()
    try:
        report = await asyncio.wait_for(cache.health(), timeout=3.0)
    except TimeoutError:
        return ComponentHealth(
            name="cache",
            status=HealthStatus.DEGRADED,
            message="Redis did not respond within 3 seconds.",
        )

    latency = (time.perf_counter() - started) * 1000
    state = report.get("status")
    if state == "healthy":
        return ComponentHealth(
            name="cache", status=HealthStatus.HEALTHY, latency_ms=round(latency, 2)
        )
    if state == "disabled":
        return ComponentHealth(
            name="cache",
            status=HealthStatus.HEALTHY,
            message="Caching is disabled by configuration.",
        )
    # A dead cache is degraded, not unhealthy: the API still returns correct
    # answers, just more slowly.
    return ComponentHealth(
        name="cache",
        status=HealthStatus.DEGRADED,
        latency_ms=round(latency, 2),
        message=f"Redis unavailable: {report.get('error', 'unknown error')}",
    )


async def _check_database() -> ComponentHealth:
    """Probe PostgreSQL with a trivial query."""
    from api.services.db_service import get_db_service

    db = get_db_service()
    started = time.perf_counter()
    try:
        ok = await asyncio.wait_for(db.ping(), timeout=3.0)
    except TimeoutError:
        return ComponentHealth(
            name="database",
            status=HealthStatus.DEGRADED,
            message="The database did not respond within 3 seconds.",
        )
    except Exception as exc:
        return ComponentHealth(
            name="database",
            status=HealthStatus.DEGRADED,
            message=f"Database unavailable: {type(exc).__name__}",
        )

    latency = (time.perf_counter() - started) * 1000
    if ok:
        return ComponentHealth(
            name="database", status=HealthStatus.HEALTHY, latency_ms=round(latency, 2)
        )
    return ComponentHealth(
        name="database",
        status=HealthStatus.DEGRADED,
        message="The database is not reachable; inference logging is paused.",
    )


def _check_models(models: Models) -> list[ComponentHealth]:
    """Report which task models are loaded and ready."""
    report = models.health()
    components: list[ComponentHealth] = []

    for task in TaskType:
        try:
            entry = models.resolve(task)
        except Exception:
            components.append(
                ComponentHealth(
                    name=f"model:{task.value}",
                    status=HealthStatus.UNHEALTHY,
                    message="No model is registered for this task.",
                )
            )
            continue

        failure = report["failures"].get(entry.key)
        loaded = any(k.startswith(f"{entry.key}:") for k in report["loaded_keys"])

        if loaded:
            state, message = HealthStatus.HEALTHY, None
        elif failure:
            state, message = HealthStatus.UNHEALTHY, f"{entry.key} failed to load: {failure[:200]}"
        else:
            # Registered but not yet loaded: fine when lazy loading is on.
            state = HealthStatus.HEALTHY if not settings.eager_model_load else HealthStatus.DEGRADED
            message = f"{entry.key} is registered but not loaded yet."

        components.append(
            ComponentHealth(name=f"model:{task.value}", status=state, message=message)
        )
    return components


def _aggregate(components: list[ComponentHealth]) -> HealthStatus:
    """Combine component states into one overall status.

    The service is unhealthy only if *every* model is unhealthy — one broken
    task should not take the other two offline.
    """
    model_components = [c for c in components if c.name.startswith("model:")]
    others = [c for c in components if not c.name.startswith("model:")]

    if model_components and all(c.status == HealthStatus.UNHEALTHY for c in model_components):
        return HealthStatus.UNHEALTHY
    if any(c.status != HealthStatus.HEALTHY for c in [*model_components, *others]):
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Full system health",
    description=(
        "Reports the status of every dependency. Returns 200 for 'healthy' and "
        "'degraded', 503 for 'unhealthy'."
    ),
)
async def health(response: Response, cache: Cache, models: Models) -> HealthResponse:
    """Check every component and report an aggregate status."""
    cache_health, db_health = await asyncio.gather(
        _check_cache(cache), _check_database(), return_exceptions=False
    )
    components = [cache_health, db_health, *_check_models(models)]
    overall = _aggregate(components)

    if overall == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment.value,
        uptime_seconds=round(uptime_seconds(), 1),
        components=components,
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description=(
        "Returns 200 whenever the process is running. Deliberately checks no "
        "dependencies, so a database blip cannot trigger a restart storm."
    ),
)
async def liveness() -> dict[str, object]:
    """Liveness probe for the orchestrator."""
    return {
        "status": "alive",
        "uptime_seconds": round(uptime_seconds(), 1),
        "version": settings.app_version,
    }


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 when this instance can serve inference requests, 503 when "
        "it cannot. Used to add or remove the instance from the load balancer."
    ),
)
async def readiness(response: Response, models: Models, inference: Inference) -> dict[str, object]:
    """Readiness probe: can this instance serve a request right now?"""
    report = models.health()
    ready = report["loaded"] > 0 or not settings.eager_model_load

    # Also refuse traffic while saturated, so the load balancer can route
    # elsewhere instead of queueing behind a full box.
    saturated = inference.inflight >= settings.max_concurrent_inferences

    if not ready or saturated:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if (ready and not saturated) else "not_ready",
        "models_loaded": report["loaded"],
        "models_registered": report["registered"],
        "inflight": inference.inflight,
        "saturated": saturated,
    }


__all__ = ["mark_started", "router", "uptime_seconds"]
