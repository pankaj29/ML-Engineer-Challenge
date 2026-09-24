"""Model metadata endpoints.

Plain English:
    "What models do you have, which version is live, and how good are they?"
    This is how a client discovers what it can pin to, and how an operator
    confirms which weights are actually serving traffic.

The metrics and limitations returned here come from the model cards, so the
honest caveats travel with the model instead of living only in a document
nobody reads.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Path, Query, status

from api.dependencies import Cache, CorrelationId, CurrentPrincipal, Models
from api.exceptions import ModelNotFoundError
from api.logging_config import get_logger
from api.models.responses import (
    ErrorResponse,
    ModelDescriptor,
    ModelMetrics,
    ModelsResponse,
)
from api.models.schemas import RuntimeFormat, TaskType

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


def _describe(entry, models: Models, defaults: dict[str, str]) -> ModelDescriptor:
    """Turn a registry entry into the public API representation."""
    report = models.health()
    loaded = any(k.startswith(f"{entry.key}:") for k in report["loaded_keys"])

    # The registry stores metrics as a flat dict. Only keys the response
    # schema declares are surfaced, so an experimental metric cannot leak out
    # of the registry into the public API. Anything dropped is logged rather
    # than discarded quietly: a metric that vanishes because of a name
    # mismatch looks identical to a metric nobody measured.
    known = {k: v for k, v in entry.metrics.items() if k in ModelMetrics.model_fields}
    dropped = sorted(set(entry.metrics) - set(known))
    if dropped:
        logger.warning(
            "model_metrics_not_in_schema",
            extra={"model": entry.key, "dropped": dropped},
        )
    metrics = ModelMetrics(**known)

    runtime = RuntimeFormat.ONNX
    for candidate in ("onnx", "torch", "tensorrt", "onnx_int8", "torch_int8"):
        if candidate in entry.artifacts:
            runtime = RuntimeFormat(candidate)
            break

    # Every format with an artifact on disk is selectable per request.
    available = [
        fmt
        for fmt in ("onnx", "onnx_int8", "torch", "torch_int8", "tensorrt")
        if fmt in entry.artifacts
    ]

    return ModelDescriptor(
        name=entry.name,
        version=entry.version,
        task=entry.task,
        runtime=runtime,
        device=report["device"],
        loaded=loaded,
        is_default=defaults.get(entry.task.value) == entry.key,
        num_classes=entry.num_classes,
        input_shape=entry.input_shape,
        available_runtimes=available,
        metrics=metrics,
        description=entry.description,
        limitations=entry.limitations,
        registered_at=(
            datetime.fromisoformat(entry.registered_at) if entry.registered_at else None
        ),
    )


@router.get(
    "",
    response_model=ModelsResponse,
    summary="List all registered models",
    description=(
        "Returns every active model with its version, task, measured metrics "
        "and known limitations, plus which version currently serves each task."
    ),
)
async def list_models(
    principal: CurrentPrincipal,
    models: Models,
    correlation_id: CorrelationId,
    task: TaskType | None = Query(default=None, description="Filter to one task."),
    loaded_only: bool = Query(default=False, description="Only models currently in memory."),
) -> ModelsResponse:
    """List registered models and their metadata."""
    defaults = models.default_keys()
    entries = models.list_entries()

    if task:
        entries = [e for e in entries if e.task == task]

    descriptors = [_describe(e, models, defaults) for e in entries]
    if loaded_only:
        descriptors = [d for d in descriptors if d.loaded]

    return ModelsResponse(
        models=descriptors,
        count=len(descriptors),
        defaults=defaults,
        correlation_id=correlation_id,
    )


@router.get(
    "/{name}",
    response_model=ModelDescriptor,
    summary="Get one model's metadata",
    responses={404: {"model": ErrorResponse, "description": "No such model or version."}},
)
async def get_model(
    principal: CurrentPrincipal,
    models: Models,
    name: str = Path(description="Registry model name."),
    version: str = Query(default="latest", description="Version, or 'latest'."),
) -> ModelDescriptor:
    """Get metadata for one model version."""
    defaults = models.default_keys()

    for entry in models.list_entries():
        if entry.name != name:
            continue
        if version == "latest":
            candidates = [e for e in models.list_entries() if e.name == name]
            entry = max(candidates, key=lambda e: e.version_tuple())
            return _describe(entry, models, defaults)
        if entry.version == version:
            return _describe(entry, models, defaults)

    raise ModelNotFoundError(
        f"No model named '{name}' with version '{version}' is registered.",
        details={
            "requested": f"{name}:{version}",
            "available": sorted({e.key for e in models.list_entries()}),
        },
    )


@router.post(
    "/reload",
    status_code=status.HTTP_200_OK,
    summary="Reload the model registry from disk",
    description=(
        "Re-reads registry.json so a newly registered model version becomes "
        "available without restarting the service. Requires the 'admin' scope."
    ),
    responses={403: {"model": ErrorResponse, "description": "Admin scope required."}},
)
async def reload_registry(
    principal: CurrentPrincipal,
    models: Models,
    cache: Cache,
    correlation_id: CorrelationId,
) -> dict[str, object]:
    """Reload the registry and invalidate cached results for changed models.

    Cache invalidation is the important half: without it, predictions made by
    the *old* weights would keep being served from Redis after a rollback.
    """
    principal.require_scope("admin")

    before = {e.key for e in models.list_entries()}
    models.reload_registry()
    after = {e.key for e in models.list_entries()}

    added = sorted(after - before)
    removed = sorted(before - after)

    invalidated = 0
    for key in removed:
        invalidated += await cache.invalidate_model(key)

    logger.info(
        "registry_reloaded",
        extra={"added": added, "removed": removed, "cache_keys_invalidated": invalidated},
    )

    return {
        "reloaded": True,
        "models_added": added,
        "models_removed": removed,
        "cache_keys_invalidated": invalidated,
        "total_models": len(after),
        "correlation_id": correlation_id,
    }


__all__ = ["router"]
