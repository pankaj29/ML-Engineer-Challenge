"""Image similarity search endpoints.

Plain English:
    "Here is a picture. Show me the ones in my collection that look like it."

    The flow has two halves. First you *index* images: each is turned into an
    embedding vector and stored. Then you *search*: the query image is turned
    into a vector the same way, and the closest stored vectors are returned.

Three endpoints:

* ``POST /similarity/embed`` — just give me the vector, I will store it myself.
* ``POST /similarity/index`` — embed this image and add it to the index.
* ``POST /similarity/search`` — embed this image and find its neighbours.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from api.dependencies import CorrelationId, CurrentPrincipal, Inference, resolve_image_bytes
from api.logging_config import get_logger
from api.middleware.monitoring import record_inference
from api.models.responses import (
    EmbeddingResponse,
    ErrorResponse,
    IndexResponse,
    SimilarHit,
    SimilarityResponse,
    TimingInfo,
)
from api.models.schemas import (
    EmbeddingRequest,
    IndexImageRequest,
    SimilarityRequest,
)
from api.services.similarity_index import SimilarityIndex, get_similarity_index

logger = get_logger(__name__)

router = APIRouter(prefix="/similarity", tags=["Similarity"])

Index = Annotated[SimilarityIndex, Depends(get_similarity_index)]

_ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "The image could not be decoded."},
    401: {"model": ErrorResponse, "description": "Missing or invalid credentials."},
    413: {"model": ErrorResponse, "description": "The image exceeds the size limit."},
    415: {"model": ErrorResponse, "description": "Unsupported image format."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded for your tier."},
    503: {"model": ErrorResponse, "description": "The embedding model is not ready."},
}


@router.post(
    "/embed",
    response_model=EmbeddingResponse,
    summary="Compute an image embedding",
    responses=_ERROR_RESPONSES,
)
async def embed(
    body: EmbeddingRequest,
    principal: CurrentPrincipal,
    inference: Inference,
) -> EmbeddingResponse:
    """Turn an image into an embedding vector without storing anything.

    Use this when you keep your own vector store. The returned vector is
    normalised to unit length, so the cosine similarity between two of them is
    simply their dot product.
    """
    image_bytes = await resolve_image_bytes(payload=body, field_name="image")

    started = time.perf_counter()
    response = await inference.embed(
        image_bytes,
        model_name=body.model_name,
        model_version=body.model_version,
        runtime=body.runtime,
        image_id=body.image_id,
    )
    record_inference(
        task="similarity",
        model=response.model.name,
        version=response.model.version,
        runtime=response.model.runtime.value,
        duration_seconds=time.perf_counter() - started,
    )
    return response


@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an image to the similarity index",
    responses=_ERROR_RESPONSES,
)
async def index_image(
    body: IndexImageRequest,
    principal: CurrentPrincipal,
    inference: Inference,
    index: Index,
    correlation_id: CorrelationId,
) -> IndexResponse:
    """Embed an image and store it so future searches can find it.

    Attach a ``label`` and arbitrary ``metadata``; both are returned with any
    search hit, so the caller does not need a second lookup to find out what
    was matched.
    """
    image_bytes = await resolve_image_bytes(payload=body, field_name="image")

    vector = await inference.embed_array(
        image_bytes, model_name=body.model_name, model_version=body.model_version
    )

    # The index sizes itself from the first vector it sees, so one embedding
    # model can be swapped for another with a different dimensionality
    # without editing configuration.
    if index.size == 0 and index.dimension != vector.shape[0]:
        index.dimension = int(vector.shape[0])
        index.clear()

    item_id = index.add(vector, item_id=body.image_id, label=body.label, metadata=body.metadata)
    logger.info(
        "image_indexed",
        extra={"item_id": item_id, "index_size": index.size, "label": body.label},
    )

    return IndexResponse(id=item_id, index_size=index.size, correlation_id=correlation_id)


@router.post(
    "/search",
    response_model=SimilarityResponse,
    summary="Find visually similar images",
    responses=_ERROR_RESPONSES,
)
async def search(
    body: SimilarityRequest,
    principal: CurrentPrincipal,
    inference: Inference,
    index: Index,
) -> SimilarityResponse:
    """Find the indexed images most similar to the supplied image.

    Scores are cosine similarities: ``1.0`` is an identical direction, ``0.0``
    unrelated, ``-1.0`` opposite. In practice, near-duplicates score above
    0.95 and "same kind of thing" lands somewhere around 0.7-0.9 — the right
    ``min_similarity`` depends on your images and is worth tuning on real data
    rather than guessing.

    An empty ``results`` list with ``index_size: 0`` means nothing has been
    indexed yet, not that nothing matched.
    """
    total_started = time.perf_counter()
    image_bytes = await resolve_image_bytes(payload=body, field_name="image")

    embed_response = await inference.embed(
        image_bytes,
        model_name=body.model_name,
        model_version=body.model_version,
        runtime=body.runtime,
        image_id=body.image_id,
    )
    vector = embed_response.embedding

    search_started = time.perf_counter()
    if index.size == 0:
        hits: list[SimilarHit] = []
    else:
        import numpy as np

        raw_hits = index.search(
            np.asarray(vector, dtype="float32"),
            top_k=body.top_k,
            min_similarity=body.min_similarity,
        )
        hits = [
            SimilarHit(
                id=item.id,
                score=round(score, 6),
                rank=rank,
                label=item.label,
                metadata=item.metadata,
            )
            for rank, (item, score) in enumerate(raw_hits, start=1)
        ]
    search_ms = (time.perf_counter() - search_started) * 1000

    record_inference(
        task="similarity",
        model=embed_response.model.name,
        version=embed_response.model.version,
        runtime=embed_response.model.runtime.value,
        duration_seconds=time.perf_counter() - total_started,
    )

    return SimilarityResponse(
        results=hits,
        count=len(hits),
        index_size=index.size,
        embedding=vector if body.include_embedding else None,
        model=embed_response.model,
        timing=TimingInfo(
            preprocess_ms=embed_response.timing.preprocess_ms,
            inference_ms=embed_response.timing.inference_ms,
            # Vector search time is reported as postprocessing, so the caller
            # can see how much of the latency is search rather than the model.
            postprocess_ms=round(search_ms, 2),
            total_ms=round((time.perf_counter() - total_started) * 1000, 2),
        ),
        correlation_id=embed_response.correlation_id,
        image_id=body.image_id,
        degraded=embed_response.degraded,
        warnings=embed_response.warnings,
    )


@router.post(
    "/upload",
    response_model=SimilarityResponse,
    summary="Find similar images for an uploaded file (multipart)",
    responses=_ERROR_RESPONSES,
)
async def search_upload(
    principal: CurrentPrincipal,
    inference: Inference,
    index: Index,
    file: Annotated[UploadFile, File(description="Query image.")],
    top_k: Annotated[int, Form(ge=1, le=100)] = 10,
    min_similarity: Annotated[float, Form(ge=-1.0, le=1.0)] = 0.0,
) -> SimilarityResponse:
    """Find similar images for a multipart file upload."""
    import numpy as np

    total_started = time.perf_counter()
    image_bytes = await resolve_image_bytes(upload=file, field_name="file")

    embed_response = await inference.embed(image_bytes, image_id=file.filename)

    search_started = time.perf_counter()
    raw_hits = (
        index.search(
            np.asarray(embed_response.embedding, dtype="float32"),
            top_k=top_k,
            min_similarity=min_similarity,
        )
        if index.size
        else []
    )
    search_ms = (time.perf_counter() - search_started) * 1000

    return SimilarityResponse(
        results=[
            SimilarHit(
                id=item.id,
                score=round(score, 6),
                rank=rank,
                label=item.label,
                metadata=item.metadata,
            )
            for rank, (item, score) in enumerate(raw_hits, start=1)
        ],
        count=len(raw_hits),
        index_size=index.size,
        model=embed_response.model,
        timing=TimingInfo(
            preprocess_ms=embed_response.timing.preprocess_ms,
            inference_ms=embed_response.timing.inference_ms,
            postprocess_ms=round(search_ms, 2),
            total_ms=round((time.perf_counter() - total_started) * 1000, 2),
        ),
        correlation_id=embed_response.correlation_id,
        image_id=file.filename,
        degraded=embed_response.degraded,
        warnings=embed_response.warnings,
    )


@router.get(
    "/stats",
    summary="Similarity index statistics",
    description="How many vectors are indexed, their dimensionality and memory use.",
)
async def index_stats(principal: CurrentPrincipal, index: Index) -> dict:
    """Return index statistics."""
    return index.stats()


__all__ = ["router"]
