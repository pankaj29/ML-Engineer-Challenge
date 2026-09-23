"""Prometheus metrics endpoint.

Plain English:
    Prometheus works by *pulling*: every 15 seconds or so it calls this
    endpoint and stores whatever numbers it finds. The response is a plain
    text format, one metric per line.

The endpoint is deliberately unauthenticated so Prometheus can scrape it
without credentials, exactly like the health checks. In production it should
not be reachable from the public internet: the Nginx gateway in this project
only allows it from inside the Docker network. Metrics are aggregate counts,
never user data or image content.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST

from api.middleware.monitoring import render_metrics

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description=(
        "Exposes counters and histograms in Prometheus text format: request "
        "rates, latency histograms, inference counts, cache hit rates and "
        "error totals."
    ),
    response_class=Response,
    responses={
        200: {
            "content": {"text/plain": {"example": 'http_requests_total{method="POST"} 42.0'}},
            "description": "Metrics in Prometheus exposition format.",
        }
    },
)
async def metrics() -> Response:
    """Render all registered metrics for a Prometheus scrape."""
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)


__all__ = ["router"]
