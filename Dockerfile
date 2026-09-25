# =============================================================================
# ML API service image - THE MAIN APPLICATION CONTAINER.
#
# This one lives at the repository root because the challenge brief's
# prescribed layout names `Dockerfile  # Main application container` there.
# The supporting services keep their images under docker/:
#     docker/Dockerfile.worker   Celery worker
#     docker/nginx/Dockerfile    API gateway
#
# Plain English:
#   A Dockerfile is a recipe for building a container image. This one uses a
#   "multi-stage build": the first stage installs everything (including the
#   compilers needed to build some Python packages), and the second stage
#   copies across only the finished result.
#
#   Why bother? The build stage needs gcc, header files and a package cache —
#   several hundred megabytes that the running service never uses. Keeping
#   them out of the final image makes it smaller (faster to pull, cheaper to
#   store) and safer: a compiler inside a production container is a useful
#   tool for an attacker who gets in.
#
# Layer ordering is deliberate. Docker caches each step and reuses it until
# something changes. Dependencies are installed BEFORE application code is
# copied, so editing a Python file rebuilds only the last few layers instead
# of reinstalling PyTorch every time.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder — compile and install dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Pinned to a specific minor version: "latest" makes builds unreproducible,
# and a base image that changes underneath you is a genuine outage source.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-only tools. These never reach the final image.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# A virtual environment makes the "copy only what is needed" step trivial:
# one self-contained directory holds every installed package.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy ONLY the requirements first. This layer is cached and reused on every
# rebuild where the dependencies have not changed — which is almost every one.
COPY requirements.txt .

# CPU-only PyTorch. The default wheel bundles CUDA libraries that add roughly
# 2 GB to the image and are useless without a GPU. GPU deployments use
# Dockerfile.api with --build-arg TORCH_INDEX set to the CUDA index instead.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip setuptools wheel \
    && pip install --extra-index-url ${TORCH_INDEX} -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: runtime — the image that actually ships
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Thread counts are pinned to 1 and concurrency is handled by running more
    # containers. Left unset, every numerical library spawns a thread per host
    # core — inside a container limited to 2 CPUs, that means dozens of threads
    # fighting over 2 cores, which is slower than a single thread.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Runtime libraries only. libgomp is required by ONNX Runtime and PyTorch;
# curl is used by the container healthcheck below.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# SECURITY: create an unprivileged user and run as it.
# A container process running as root that escapes its namespace is root on
# the host. Running as an ordinary user removes that entire class of
# escalation, and costs nothing.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

WORKDIR /app

# Application code, copied per-directory so an edit to one does not invalidate
# the cached layers of the others.
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser db/ ./db/
COPY --chown=appuser:appuser models/registry.py ./models/registry.py
# The registry JSON, not just the code that reads it. Without this the
# service starts, reports healthy on its dependencies, and loads zero
# models, which only shows up as a 503 from /health.
COPY --chown=appuser:appuser models/registry.json ./models/registry.json
COPY --chown=appuser:appuser models/__init__.py ./models/__init__.py

# Serving does not use these. The init container and the drift CronJob do, and
# they run this same image rather than a second one: a separate image would
# drift out of step with the API it is meant to be checking.
#
# All three are pure Python over numpy and the database, so they add kilobytes
# rather than the training stack.
COPY --chown=appuser:appuser models/validation/ ./models/validation/
COPY --chown=appuser:appuser models/pipeline/ ./models/pipeline/
COPY --chown=appuser:appuser scripts/fetch_artifacts.py ./scripts/fetch_artifacts.py
COPY --chown=appuser:appuser models/artifacts_manifest.json ./models/artifacts_manifest.json

# Only the Celery *app* (broker configuration), never worker/tasks.py. The API
# enqueues jobs by task name via send_task and reads their status, so it needs
# the broker settings but not the worker's implementation — which would drag
# the whole model-loading stack into this image for no reason.
COPY --chown=appuser:appuser worker/__init__.py ./worker/__init__.py
COPY --chown=appuser:appuser worker/celery_app.py ./worker/celery_app.py

# Mount points for model artifacts and logs. Artifacts are mounted as a volume
# rather than baked in: a 100 MB model inside the image means rebuilding and
# redeploying the whole service to ship new weights.
RUN mkdir -p /app/models/artifacts /app/data /app/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# The healthcheck uses the liveness probe, which deliberately checks no
# dependencies — see api/routers/health.py. Using the full health endpoint
# here would restart every container whenever the database hiccuped.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/api/v1/health/live || exit 1

# exec form (JSON array), not shell form: it makes uvicorn PID 1, so it
# receives SIGTERM directly and can shut down gracefully. In shell form a
# shell would be PID 1 and would not forward the signal, so every deploy
# would kill in-flight requests.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log", \
     "--timeout-graceful-shutdown", "30"]

# Note on --no-access-log: MonitoringMiddleware already emits a richer,
# structured access record per request, carrying the correlation id and the
# timing breakdown. Uvicorn's own line would be a duplicate in a different
# format.
#
# An earlier version passed `--log-config /dev/null` to suppress uvicorn's
# logger setup. That fails: uvicorn tries to *parse* the path as a logging
# config file and exits with "/dev/null is an empty file". No log-config
# argument is needed, because the application configures its own logging in
# the lifespan handler.
