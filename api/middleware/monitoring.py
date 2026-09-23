"""Request logging and Prometheus metrics.

Plain English:
    This middleware watches every request and records two kinds of evidence:

    * **Logs** — one structured line per request, carrying the correlation id,
      so you can reconstruct what happened to a specific user.
    * **Metrics** — counters and histograms that Prometheus scrapes, so you
      can see patterns across *all* requests: error rate, throughput, p95
      latency.

    Logs answer "what happened to this request?". Metrics answer "is the
    system healthy right now?". You need both.

A deliberate detail: metric **labels** use the route *template*
(``/api/v1/batch/{job_id}``) and never the actual path
(``/api/v1/batch/abc-123``). Putting an unbounded value like a job id into a
label creates a new time series per job — "cardinality explosion" — which is
the standard way teams accidentally take down their own Prometheus.

Histograms are used rather than plain averages because an average latency
hides the tail, and the tail is what users feel.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import Request
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from api.config import Settings, settings
from api.logging_config import bind_correlation_id, get_logger

logger = get_logger(__name__)

# A dedicated registry keeps our metrics separate from any library that
# registers into the global default, which makes tests deterministic.
REGISTRY = CollectorRegistry()

# Buckets chosen around the sub-second requirement in the brief: dense
# resolution below 1 s, then coarse buckets to catch pathological outliers.
_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
)

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests processed.",
    ["method", "endpoint", "status_code", "tier"],
    registry=REGISTRY,
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "End-to-end HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently being handled.",
    ["method", "endpoint"],
    registry=REGISTRY,
)
http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "Uploaded request body size in bytes.",
    ["endpoint"],
    buckets=(1024, 10240, 102400, 512000, 1048576, 5242880, 10485760),
    registry=REGISTRY,
)

inference_total = Counter(
    "inference_total",
    "Model inferences performed.",
    ["task", "model", "version", "runtime", "status"],
    registry=REGISTRY,
)
inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Model forward-pass latency in seconds, excluding HTTP overhead.",
    ["task", "model", "runtime"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
inference_in_progress = Gauge(
    "inference_in_progress",
    "Inferences executing right now.",
    registry=REGISTRY,
)

cache_operations_total = Counter(
    "cache_operations_total",
    "Cache lookups by outcome.",
    ["operation", "result"],
    registry=REGISTRY,
)

model_load_total = Counter(
    "model_load_total",
    "Model load attempts.",
    ["model", "runtime", "status"],
    registry=REGISTRY,
)
models_loaded = Gauge(
    "models_loaded",
    "Models currently resident in memory.",
    registry=REGISTRY,
)

rate_limit_rejections_total = Counter(
    "rate_limit_rejections_total",
    "Requests rejected by the rate limiter.",
    ["tier"],
    registry=REGISTRY,
)

errors_total = Counter(
    "errors_total",
    "Errors returned to clients, by error code.",
    ["code", "endpoint"],
    registry=REGISTRY,
)

batch_jobs_total = Counter(
    "batch_jobs_total",
    "Background batch jobs by terminal status.",
    ["task", "status"],
    registry=REGISTRY,
)
batch_job_duration_seconds = Histogram(
    "batch_job_duration_seconds",
    "Batch job wall-clock duration in seconds.",
    ["task"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 900),
    registry=REGISTRY,
)

app_info = Gauge(
    "app_info",
    "Application build information (always 1; the labels carry the data).",
    ["version", "environment"],
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Render the registry in Prometheus text format.

    When ``PROMETHEUS_MULTIPROC_DIR`` is set (several uvicorn workers in one
    container), metrics are collected from every worker's shared files so the
    scrape reflects the whole container rather than whichever worker answered.
    """
    import os

    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(REGISTRY)


def route_template(request: Request) -> str:
    """Return the matched route pattern, not the concrete path.

    ``/api/v1/batch/{job_id}`` rather than ``/api/v1/batch/9f2c...``. See the
    module docstring for why this matters.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return str(path)
    # No route matched (a 404). Bucket them all together rather than emitting
    # a distinct label for every probed URL.
    return "unmatched"


class MonitoringMiddleware:
    """Logs and measures every request.

    Implemented as raw ASGI so that the correlation id set here propagates
    into route handlers through ``contextvars``. Starlette's
    ``BaseHTTPMiddleware`` runs handlers in a separate task and would break
    that propagation.
    """

    def __init__(self, app: Any, config: Settings | None = None) -> None:
        self.app = app
        self.settings = config or settings

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope.setdefault("state", {})
        request = Request(scope, receive=receive)

        # Reuse an inbound correlation id when a caller or upstream proxy
        # supplies one, so a trace spans several services.
        incoming = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID")
        correlation_id = bind_correlation_id(incoming)
        scope["state"]["correlation_id"] = correlation_id

        method = request.method
        started = time.perf_counter()
        status_code = 500
        endpoint = "unmatched"

        content_length = request.headers.get("content-length")
        request_bytes = int(content_length) if content_length and content_length.isdigit() else 0

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.setdefault("headers", [])
                headers.append((b"x-correlation-id", correlation_id.encode()))
            await send(message)

        in_progress_tracked = False
        try:
            # The route is only known after routing, so track in-progress
            # under the raw path and re-label once the response is done.
            http_requests_in_progress.labels(method=method, endpoint="all").inc()
            in_progress_tracked = True
            await self.app(scope, receive, send_wrapper)
        except Exception:
            status_code = 500
            raise
        finally:
            if in_progress_tracked:
                http_requests_in_progress.labels(method=method, endpoint="all").dec()

            duration = time.perf_counter() - started
            endpoint = route_template(request)
            principal = scope.get("state", {}).get("principal")
            tier = getattr(getattr(principal, "tier", None), "value", "anonymous")

            if self.settings.metrics_enabled:
                http_requests_total.labels(
                    method=method, endpoint=endpoint, status_code=str(status_code), tier=tier
                ).inc()
                http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
                    duration
                )
                if request_bytes:
                    http_request_size_bytes.labels(endpoint=endpoint).observe(request_bytes)
                if status_code >= 400:
                    errors_total.labels(code=str(status_code), endpoint=endpoint).inc()
                if status_code == 429:
                    rate_limit_rejections_total.labels(tier=tier).inc()

            # Health and metrics probes fire constantly; logging them at INFO
            # would bury real traffic. They are logged at DEBUG instead.
            is_probe = endpoint.endswith(("/health", "/metrics", "/health/live", "/health/ready"))
            slow = duration > self.settings.slow_request_threshold_seconds

            level = logger.debug
            if not is_probe:
                level = logger.info
            if status_code >= 500:
                level = logger.error
            elif status_code >= 400 or slow:
                level = logger.warning

            level(
                "http_request",
                extra={
                    "method": method,
                    "path": request.url.path,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "request_bytes": request_bytes,
                    "tier": tier,
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent", "")[:120],
                    "slow": slow or None,
                },
            )


def record_inference(
    *,
    task: str,
    model: str,
    version: str,
    runtime: str,
    duration_seconds: float,
    status: str = "success",
) -> None:
    """Record one model inference in the metrics registry.

    Called by the routers after a prediction so that inference latency is
    tracked separately from total HTTP latency. The gap between the two is
    time spent on validation, serialisation and network — useful to know when
    total latency rises but the model has not got slower.
    """
    if not settings.metrics_enabled:
        return
    inference_total.labels(
        task=task, model=model, version=version, runtime=runtime, status=status
    ).inc()
    if status == "success":
        inference_duration_seconds.labels(task=task, model=model, runtime=runtime).observe(
            duration_seconds
        )


def record_cache(operation: str, result: str) -> None:
    """Record a cache hit, miss or error."""
    if settings.metrics_enabled:
        cache_operations_total.labels(operation=operation, result=result).inc()


def record_model_load(model: str, runtime: str, status: str) -> None:
    """Record a model load attempt."""
    if settings.metrics_enabled:
        model_load_total.labels(model=model, runtime=runtime, status=status).inc()


def set_app_info(version: str, environment: str) -> None:
    """Publish build information as a labelled gauge, set once at startup."""
    app_info.labels(version=version, environment=environment).set(1)


__all__ = [
    "REGISTRY",
    "MonitoringMiddleware",
    "batch_job_duration_seconds",
    "batch_jobs_total",
    "cache_operations_total",
    "errors_total",
    "http_request_duration_seconds",
    "http_requests_total",
    "inference_duration_seconds",
    "inference_in_progress",
    "inference_total",
    "models_loaded",
    "rate_limit_rejections_total",
    "record_cache",
    "record_inference",
    "record_model_load",
    "render_metrics",
    "route_template",
    "set_app_info",
]
