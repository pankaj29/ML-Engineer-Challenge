"""Single source of truth for the deliverables checklist.

Every row below is a requirement lifted directly from README.md (the challenge
brief). ``status`` is updated as work lands, and
``scripts/generate_checklist.py`` renders this list into
``DELIVERABLES_CHECKLIST.xlsx``.

Keeping the data in Python rather than editing the spreadsheet by hand means
the checklist can never drift out of sync: regenerate it and it is correct.

Status vocabulary:
    DONE        - delivered and verified with evidence (a test run, output, file)
    IN PROGRESS - actively being built
    TODO        - not started
    BLOCKED     - cannot be completed; the reason is recorded in Notes
"""

from __future__ import annotations

from typing import NamedTuple


class Item(NamedTuple):
    """One checklist row."""

    part: str  # which part of the brief this belongs to
    requirement: str  # the requirement, quoted or closely paraphrased from README
    status: str  # DONE / IN PROGRESS / TODO / BLOCKED
    evidence: str  # file path, command, or artifact proving it is done
    notes: str  # assumptions, caveats, decisions


ITEMS: list[Item] = [
    # ================================================================
    # Part 1 - Model Development & Optimisation
    # ================================================================
    Item(
        "Part 1: Models",
        "Use 3 different models (classification, detection, + similarity search)",
        "TODO",
        "models/",
        "README overview names classification, detection AND similarity search; "
        "the numbered list names only two. Confirmed with user: similarity is the 3rd model.",
    ),
    Item(
        "Part 1: Models",
        "Image classification model (ViT / ResNet / EfficientNet)",
        "TODO",
        "models/training/train_classifier.py",
        "",
    ),
    Item(
        "Part 1: Models",
        "Object detection model (YOLO / DETR) on COCO subset",
        "TODO",
        "models/detection/",
        "",
    ),
    Item(
        "Part 1: Models",
        "Fine-tune classifier with MIXED PRECISION",
        "TODO",
        "models/training/train_classifier.py",
        "torch.amp autocast + GradScaler; GPU required for fp16.",
    ),
    Item(
        "Part 1: Models",
        "Fine-tune classifier with GRADIENT CLIPPING",
        "TODO",
        "models/training/train_classifier.py",
        "",
    ),
    Item(
        "Part 1: Models",
        "Fine-tune classifier with LEARNING RATE SCHEDULING",
        "TODO",
        "models/training/train_classifier.py",
        "",
    ),
    Item(
        "Part 1: Models",
        "Use the tiny-ImageNet dataset for classification fine-tuning",
        "TODO",
        "scripts/download_datasets.py",
        "",
    ),
    Item(
        "Part 1: Models",
        "Implement custom data augmentation pipeline",
        "TODO",
        "models/training/augmentation.py",
        "",
    ),
    Item(
        "Part 1: Optimisation",
        "Apply INT8 quantization to all models",
        "TODO",
        "models/optimization/quantize.py",
        "",
    ),
    Item(
        "Part 1: Optimisation",
        "Convert models to ONNX format",
        "TODO",
        "models/optimization/export_onnx.py",
        "",
    ),
    Item(
        "Part 1: Optimisation",
        "Convert models to TensorRT format",
        "TODO",
        "models/optimization/export_tensorrt.py",
        "Built and served on an A100.",
    ),
    Item(
        "Part 1: Optimisation",
        "Benchmark inference times across all formats",
        "TODO",
        "benchmarks/",
        "",
    ),
    Item(
        "Part 1: Validation",
        "Comprehensive model validation pipeline",
        "TODO",
        "models/validation/validate.py",
        "",
    ),
    Item(
        "Part 1: Validation",
        "A/B testing framework for model comparison",
        "TODO",
        "models/validation/ab_test.py",
        "",
    ),
    Item(
        "Part 1: Validation",
        "Model drift detection using statistical tests",
        "TODO",
        "models/validation/drift.py",
        "KS test + chi-square + PSI.",
    ),
    Item(
        "Part 1: Validation",
        "Performance regression testing",
        "TODO",
        "models/validation/regression.py",
        "",
    ),
    Item(
        "Part 1: Deliverable",
        "models/ directory with training scripts and model artifacts",
        "TODO",
        "models/",
        "",
    ),
    Item(
        "Part 1: Deliverable",
        "benchmarks/ directory with performance comparison reports",
        "TODO",
        "benchmarks/reports/",
        "",
    ),
    Item(
        "Part 1: Deliverable",
        "Comprehensive model cards with metrics and limitations",
        "TODO",
        "models/cards/",
        "",
    ),
    # ================================================================
    # Part 2 - Production API Development
    # ================================================================
    Item(
        "Part 2: Structure",
        "FastAPI app structure exactly as specified in README",
        "IN PROGRESS",
        "api/",
        "routers/, models/, services/, middleware/, utils/ all present.",
    ),
    Item("Part 2: Structure", "api/main.py", "TODO", "api/main.py", ""),
    Item(
        "Part 2: Structure",
        "api/routers/classification.py",
        "TODO",
        "api/routers/classification.py",
        "",
    ),
    Item("Part 2: Structure", "api/routers/detection.py", "TODO", "api/routers/detection.py", ""),
    Item(
        "Part 2: Structure",
        "api/models/schemas.py",
        "DONE",
        "api/models/schemas.py",
        "Request schemas + shared enums.",
    ),
    Item(
        "Part 2: Structure",
        "api/models/responses.py",
        "DONE",
        "api/models/responses.py",
        "Response schemas for every endpoint.",
    ),
    Item(
        "Part 2: Structure",
        "api/services/model_service.py",
        "TODO",
        "api/services/model_service.py",
        "",
    ),
    Item(
        "Part 2: Structure",
        "api/services/cache_service.py",
        "TODO",
        "api/services/cache_service.py",
        "",
    ),
    Item(
        "Part 2: Structure",
        "api/services/inference_service.py",
        "TODO",
        "api/services/inference_service.py",
        "",
    ),
    Item("Part 2: Structure", "api/middleware/auth.py", "TODO", "api/middleware/auth.py", ""),
    Item(
        "Part 2: Structure",
        "api/middleware/rate_limit.py",
        "TODO",
        "api/middleware/rate_limit.py",
        "",
    ),
    Item(
        "Part 2: Structure",
        "api/middleware/monitoring.py",
        "TODO",
        "api/middleware/monitoring.py",
        "",
    ),
    Item(
        "Part 2: Structure",
        "api/utils/image_processing.py",
        "DONE",
        "api/utils/image_processing.py",
        "Decode, EXIF fix, centre-crop + letterbox resize, normalise, box un-letterboxing.",
    ),
    Item(
        "Part 2: Structure",
        "api/utils/validators.py",
        "DONE",
        "api/utils/validators.py",
        "Size/format/dimension/bomb checks + SSRF guard on image_url.",
    ),
    Item("Part 2: Endpoints", "POST /api/v1/classify", "TODO", "api/routers/classification.py", ""),
    Item("Part 2: Endpoints", "POST /api/v1/detect", "TODO", "api/routers/detection.py", ""),
    Item("Part 2: Endpoints", "POST /api/v1/batch", "TODO", "api/routers/batch.py", ""),
    Item("Part 2: Endpoints", "GET /api/v1/models", "TODO", "api/routers/models.py", ""),
    Item("Part 2: Endpoints", "GET /api/v1/health", "TODO", "api/routers/health.py", ""),
    Item(
        "Part 2: Endpoints",
        "GET /api/v1/metrics (Prometheus)",
        "TODO",
        "api/routers/metrics.py",
        "",
    ),
    Item(
        "Part 2: Features",
        "Rate limiting with different limits per user tier",
        "TODO",
        "api/middleware/rate_limit.py",
        "free/basic/pro/enterprise tiers.",
    ),
    Item(
        "Part 2: Features",
        "Comprehensive image validation (size, format, content)",
        "DONE",
        "api/utils/validators.py",
        "Verified: empty, garbage, truncated, disallowed format, bomb, SSRF all rejected.",
    ),
    Item(
        "Part 2: Features",
        "Async processing: background jobs for large batch requests",
        "TODO",
        "worker/",
        "Celery worker.",
    ),
    Item(
        "Part 2: Features",
        "Model versioning: support multiple model versions",
        "TODO",
        "api/services/model_service.py",
        "",
    ),
    Item(
        "Part 2: Features",
        "Graceful degradation: fallback when models fail",
        "TODO",
        "api/services/inference_service.py",
        "",
    ),
    Item(
        "Part 2: Errors",
        "Structured logging with correlation IDs",
        "DONE",
        "api/logging_config.py",
        "JSON logs, contextvar-based correlation id. Verified with sample output.",
    ),
    Item(
        "Part 2: Errors",
        "Custom exception handling with user-friendly messages",
        "DONE",
        "api/exceptions.py",
        "Single error envelope; internal detail logged, never returned.",
    ),
    Item(
        "Part 2: Errors",
        "Request/response logging for debugging",
        "TODO",
        "api/middleware/monitoring.py",
        "",
    ),
    Item(
        "Part 2: Errors",
        "Performance monitoring and alerting",
        "TODO",
        "monitoring/",
        "",
    ),
    Item(
        "Part 2: Deliverable",
        "Complete FastAPI application with all endpoints",
        "TODO",
        "api/main.py",
        "",
    ),
    Item(
        "Part 2: Deliverable",
        "Comprehensive API documentation (auto-generated + custom)",
        "TODO",
        "docs/API.md",
        "",
    ),
    Item(
        "Part 2: Deliverable",
        "Postman collection or OpenAPI spec for testing",
        "TODO",
        "docs/openapi.json",
        "",
    ),
    # ================================================================
    # Part 3 - Testing
    # ================================================================
    Item("Part 3: Unit", "Unit tests, target >90% coverage", "TODO", "tests/unit/", ""),
    Item("Part 3: Unit", "Test model inference functions", "TODO", "tests/unit/", ""),
    Item(
        "Part 3: Unit",
        "Test image preprocessing utilities",
        "TODO",
        "tests/unit/test_image_processing.py",
        "",
    ),
    Item("Part 3: Unit", "Test API route handlers", "TODO", "tests/unit/", ""),
    Item("Part 3: Unit", "Test service layer functions", "TODO", "tests/unit/", ""),
    Item(
        "Part 3: Unit",
        "Mock external dependencies",
        "TODO",
        "tests/conftest.py",
        "fakeredis + aiosqlite.",
    ),
    Item("Part 3: Integration", "Test database operations", "TODO", "tests/integration/", ""),
    Item(
        "Part 3: Integration", "Test model loading and inference", "TODO", "tests/integration/", ""
    ),
    Item("Part 3: Integration", "Test API endpoint integration", "TODO", "tests/integration/", ""),
    Item(
        "Part 3: Performance",
        "Stress testing for model inference",
        "TODO",
        "tests/performance/",
        "locust.",
    ),
    Item("Part 3: Performance", "Memory usage profiling", "TODO", "tests/performance/", ""),
    Item(
        "Part 3: Infra",
        "Pytest configuration with fixtures",
        "TODO",
        "pytest.ini / tests/conftest.py",
        "",
    ),
    Item("Part 3: Infra", "Test database setup/teardown", "TODO", "tests/conftest.py", ""),
    Item(
        "Part 3: Infra", "Mock services for external dependencies", "TODO", "tests/conftest.py", ""
    ),
    Item(
        "Part 3: Infra", "Continuous testing with GitHub Actions", "TODO", ".github/workflows/", ""
    ),
    Item("Part 3: Deliverable", "tests/ directory with complete test suite", "TODO", "tests/", ""),
    Item("Part 3: Deliverable", "Test configuration and fixtures", "TODO", "tests/conftest.py", ""),
    Item("Part 3: Deliverable", "Performance test reports", "TODO", "benchmarks/reports/", ""),
    Item(
        "Part 3: Deliverable",
        "CI/CD pipeline configuration",
        "TODO",
        ".github/workflows/ci.yml",
        "",
    ),
    # ================================================================
    # Part 4 - Containerisation & Orchestration
    # ================================================================
    Item("Part 4: Docker", "Multi-stage builds for optimal image size", "TODO", "docker/", ""),
    Item(
        "Part 4: Docker",
        "Security best practices (non-root user, minimal base)",
        "TODO",
        "docker/",
        "",
    ),
    Item("Part 4: Docker", "Optimised for layer caching", "TODO", "docker/", ""),
    Item("Part 4: Docker", "Separate images for different services", "TODO", "docker/", ""),
    Item(
        "Part 4: Compose",
        "Service: api-gateway (Nginx or Traefik)",
        "TODO",
        "docker-compose.yml",
        "",
    ),
    Item("Part 4: Compose", "Service: ml-api (FastAPI)", "TODO", "docker-compose.yml", ""),
    Item("Part 4: Compose", "Service: worker (Celery)", "TODO", "docker-compose.yml", ""),
    Item("Part 4: Compose", "Service: redis", "TODO", "docker-compose.yml", ""),
    Item("Part 4: Compose", "Service: postgres", "TODO", "docker-compose.yml", ""),
    Item("Part 4: Compose", "Service: prometheus", "TODO", "docker-compose.yml", ""),
    Item("Part 4: Compose", "Service: grafana (optional)", "TODO", "docker-compose.yml", ""),
    Item(
        "Part 4: Config",
        "Environment-based configuration",
        "DONE",
        "api/config.py + .env.example",
        "Pydantic Settings; fails fast if insecure in staging/production.",
    ),
    Item("Part 4: Config", "Health checks for all services", "TODO", "docker-compose.yml", ""),
    Item("Part 4: Config", "Proper resource limits", "TODO", "docker-compose.prod.yml", ""),
    Item(
        "Part 4: Config", "Volume management for persistent data", "TODO", "docker-compose.yml", ""
    ),
    Item(
        "Part 4: Config",
        "Network configuration and service discovery",
        "TODO",
        "docker-compose.yml",
        "",
    ),
    Item("Part 4: Deliverable", "Dockerfile for each service", "TODO", "docker/", ""),
    Item(
        "Part 4: Deliverable",
        "docker-compose.yml for local development",
        "TODO",
        "docker-compose.yml",
        "",
    ),
    Item(
        "Part 4: Deliverable",
        "docker-compose.prod.yml for production",
        "TODO",
        "docker-compose.prod.yml",
        "",
    ),
    Item(
        "Part 4: Deliverable",
        "Documentation for deployment and scaling",
        "TODO",
        "docs/DEPLOYMENT.md",
        "",
    ),
    # ================================================================
    # Submission - Documentation
    # ================================================================
    Item("Docs", "README: Architecture overview and design decisions", "TODO", "README.md", ""),
    Item("Docs", "README: Setup and installation instructions", "TODO", "README.md", ""),
    Item("Docs", "README: API usage examples", "TODO", "README.md", ""),
    Item("Docs", "README: Performance benchmarks", "TODO", "README.md", ""),
    Item("Docs", "README: Known limitations and future improvements", "TODO", "README.md", ""),
    Item("Docs", "API docs: Complete endpoint documentation", "TODO", "docs/API.md", ""),
    Item("Docs", "API docs: Request/response examples", "TODO", "docs/API.md", ""),
    Item("Docs", "API docs: Error handling guide", "TODO", "docs/API.md", ""),
    Item("Docs", "API docs: Authentication guide", "TODO", "docs/API.md", ""),
    Item(
        "Docs",
        "Technical doc: Model selection and optimisation rationale",
        "TODO",
        "docs/TECHNICAL.md",
        "",
    ),
    Item(
        "Docs", "Technical doc: Performance benchmarking results", "TODO", "docs/TECHNICAL.md", ""
    ),
    Item("Docs", "Technical doc: System architecture decisions", "TODO", "docs/TECHNICAL.md", ""),
    Item("Docs", "Technical doc: Scalability considerations", "TODO", "docs/TECHNICAL.md", ""),
    Item("Docs", "Documented assumptions", "TODO", "docs/ASSUMPTIONS.md", ""),
    # ================================================================
    # Submission - Quality standards
    # ================================================================
    Item("Quality", "PEP 8 compliance, type hints, docstrings", "TODO", "ruff + mypy in CI", ""),
    Item("Quality", "Test coverage minimum 85% for critical paths", "TODO", "pytest --cov", ""),
    Item(
        "Quality", "Performance: sub-second inference for single images", "TODO", "benchmarks/", ""
    ),
    Item(
        "Quality",
        "Security: no hardcoded secrets, proper input validation",
        "TODO",
        "api/config.py, api/utils/validators.py",
        "",
    ),
    Item("Quality", "Scalability: design for horizontal scaling", "TODO", "docs/TECHNICAL.md", ""),
]

# ---------------------------------------------------------------------------
# Status updates
# ---------------------------------------------------------------------------
# Ticking a requirement off is a one-line edit here, rather than hunting for
# the right Item() above. Each key is a unique substring of the requirement
# text; the value is (status, evidence, notes). Applied at import time.
#
# The rule for marking something DONE: there must be evidence. A command that
# was run, output that was seen, a test that passed. "The code looks right" is
# not evidence.
UPDATES: dict[str, tuple[str, str, str]] = {
    # --- Part 1: models & optimisation -----------------------------------
    "Use 3 different models": (
        "DONE",
        "models/registry.json; python -m models.registry list",
        "resnet50 (classification), yolov8n (detection), resnet50-embed (similarity). "
        "All three verified serving real predictions.",
    ),
    "Object detection model (YOLO": (
        "DONE",
        "scripts/prepare_models.py; models/validation/coco_eval.py; benchmarks/reports/coco_eval.json",
        "YOLOv8n with Ultralytics COCO weights. Measured on 500 COCO val2017 images through the serving path: 0.392 mAP50-95, 0.536 mAP50 (pycocotools).",
    ),
    "Apply INT8 quantization to all models": (
        "DONE",
        "models/optimization/quantize.py; benchmarks/reports/quantization.json; benchmarks/reports/int8_fidelity.json",
        "All 4 models, static QDQ, uint8 with percentile calibration (chosen by measurement, benchmarks/reports/int8_recipes.json): resnet50 3.92x smaller, resnet50-tiny-imagenet 3.91x, resnet50-embed 3.91x, yolov8n 3.67x (convolutions only, COCO-calibrated; 0.388 against 0.392 mAP50-95). The fine-tuned classifier loses 0.53 points of top-1 on 10,000 images. INT8 is selectable per request and the default for none; tests/integration/test_quantized_fidelity.py checks every INT8 model against fp32 in CI.",
    ),
    "Convert models to ONNX format": (
        "DONE",
        "models/optimization/export_onnx.py; scripts/verify_onnx_parity.py; benchmarks/reports/onnx_export.json",
        "All 4 exported and checked against PyTorch on real images; parity recorded for each in onnx_export.json. Uses the legacy exporter: torch 2.9's dynamo path ignored dynamic_axes and split weights into a sidecar file.",
    ),
    "api/models/schemas.py": ("DONE", "api/models/schemas.py", "Request schemas + shared enums."),
    "api/models/responses.py": (
        "DONE",
        "api/models/responses.py",
        "Response schemas for every endpoint.",
    ),
    # --- Part 2: structure -------------------------------------------------
    "FastAPI app structure exactly": (
        "DONE",
        "api/",
        "Matches the README tree. Added api/dependencies.py (shared FastAPI "
        "dependencies) and api/config.py, both documented in the README.",
    ),
    "api/main.py": (
        "DONE",
        "api/main.py",
        "App factory, lifespan, middleware stack, custom OpenAPI.",
    ),
    "api/routers/classification.py": (
        "DONE",
        "api/routers/classification.py",
        "JSON + multipart variants.",
    ),
    "api/routers/detection.py": ("DONE", "api/routers/detection.py", "JSON + multipart variants."),
    "api/services/model_service.py": (
        "DONE",
        "api/services/model_service.py",
        "Registry, version resolution, ONNX/Torch/TensorRT runtimes, cached thread-safe loading.",
    ),
    "api/services/cache_service.py": (
        "DONE",
        "api/services/cache_service.py",
        "Redis result cache; fails soft to cache-miss when Redis is down (verified).",
    ),
    "api/services/inference_service.py": (
        "DONE",
        "api/services/inference_service.py",
        "Orchestration, softmax/NMS, concurrency semaphore, timeouts, fallbacks.",
    ),
    "api/middleware/auth.py": (
        "DONE",
        "api/middleware/auth.py",
        "API key + JWT. Verified: valid key 200, bad key 401, no key 401.",
    ),
    "api/middleware/rate_limit.py": (
        "DONE",
        "api/middleware/rate_limit.py",
        "Redis Lua token bucket, per-tier. Verified: 3 rpm allows 3, blocks the 4th.",
    ),
    "api/middleware/monitoring.py": (
        "DONE",
        "api/middleware/monitoring.py",
        "Correlation IDs, request logging, Prometheus counters/histograms.",
    ),
    # --- Part 2: endpoints -------------------------------------------------
    "POST /api/v1/classify": (
        "DONE",
        "api/routers/classification.py",
        "Verified 200 with real predictions.",
    ),
    "POST /api/v1/detect": (
        "DONE",
        "api/routers/detection.py",
        "Verified live through the gateway, in both the dev and production stacks.",
    ),
    "POST /api/v1/batch": (
        "DONE",
        "api/routers/batch.py",
        "202 + job id; Celery worker; poll/cancel endpoints.",
    ),
    "GET /api/v1/models": (
        "DONE",
        "api/routers/models.py",
        "Lists every registered model with its default per task.",
    ),
    "GET /api/v1/health": (
        "DONE",
        "api/routers/health.py",
        "Plus /health/live and /health/ready probes.",
    ),
    "GET /api/v1/metrics": (
        "DONE",
        "api/routers/metrics.py",
        "Verified: Prometheus text format returned.",
    ),
    # --- Part 2: features --------------------------------------------------
    "Rate limiting with different limits per user tier": (
        "DONE",
        "api/middleware/rate_limit.py",
        "free 10 / basic 60 / pro 300 / enterprise 3000 rpm. Batch costs N tokens.",
    ),
    "Async processing": (
        "DONE",
        "worker/celery_app.py, worker/tasks.py, api/routers/batch.py",
        "Celery + Redis. One bad image fails only its own item, never the batch.",
    ),
    "Model versioning": (
        "DONE",
        "api/services/model_service.py, models/registry.py",
        "Pin model_name/model_version per request, or 'latest'. POST /models/reload picks up new versions live.",
    ),
    "Graceful degradation": (
        "DONE",
        "api/services/inference_service.py",
        "Runtime fallback chain + model fallback with degraded:true. A typo'd model name "
        "correctly returns 404 rather than silently serving a different model.",
    ),
    "Structured logging with correlation IDs": (
        "DONE",
        "api/logging_config.py",
        "JSON logs, contextvar correlation id, returned as X-Correlation-ID (verified).",
    ),
    "Custom exception handling": (
        "DONE",
        "api/exceptions.py",
        "One error envelope. Verified: 400/401/404/413/415/422/429/503 all correct shape.",
    ),
    "Request/response logging": (
        "DONE",
        "api/middleware/monitoring.py",
        "One structured line per request.",
    ),
    "Performance monitoring and alerting": (
        "DONE",
        "api/middleware/monitoring.py; monitoring/",
        "Prometheus metrics for requests, inference (failures included), cache, model loads and batch jobs. 11 alert rules validated by promtool, each on a user-visible symptom. 18-panel Grafana dashboard, auto-provisioned. Prometheus scrapes every API replica and every worker through DNS discovery, verified live with 3 replicas.",
    ),
    # --- Part 1: training & validation -------------------------------------
    "Image classification model (ViT / ResNet": (
        "DONE",
        "models/training/train_classifier.py; scripts/prepare_models.py",
        "ResNet-50 for serving (ImageNet-1k) plus a Tiny-ImageNet fine-tuning pipeline.",
    ),
    "Fine-tune classifier with MIXED PRECISION": (
        "DONE",
        "models/training/train_classifier.py; benchmarks/reports/resnet50_training_history.json",
        "torch.autocast fp16 with GradScaler on CUDA (bf16 on CPU). 60 epochs on an A100: 78.91% top-1, 92.12% top-5, identical when the exported ONNX model is scored through the serving path on all 10,000 validation images.",
    ),
    "Fine-tune classifier with GRADIENT CLIPPING": (
        "DONE",
        "models/training/train_classifier.py",
        "clip_grad_norm_ at 1.0 after unscaling (the order matters with AMP). Grad norm and loss scale logged per epoch; the end-of-epoch loss scale drops at epochs 12, 18, 25, 43 and 49, where GradScaler skipped non-finite steps.",
    ),
    "Fine-tune classifier with LEARNING RATE SCHEDULING": (
        "DONE",
        "models/training/train_classifier.py",
        "Cosine with linear warmup (default), plus onecycle/step/plateau. Per-batch vs "
        "per-epoch stepping handled explicitly. Cosine + 5% warmup over the full 60-epoch run.",
    ),
    "Use the tiny-ImageNet dataset": (
        "DONE",
        "scripts/download_datasets.py; models/training/dataset.py",
        "All 200 classes, 100,000 training and 10,000 validation images (120,203 files, 240 MB). The validation loader reads val_annotations.txt because ImageFolder labels that split as one class, a bug in the provided starter script.",
    ),
    "Implement custom data augmentation pipeline": (
        "DONE",
        "models/training/augmentation.py",
        "Hand-written RandAugment (13 ops), RandomResizedCrop, RandomErasing, MixUp, CutMix.",
    ),
    "Convert models to TensorRT format": (
        "DONE",
        "models/optimization/export_tensorrt.py; benchmarks/reports/tensorrt.json",
        "Built and verified on an A100 (TensorRT 11.3.0.99) for the fine-tuned classifier: INT8 0.920 ms p50, 1068 img/s, 24.1 MB; fp16 0.990 ms; fp32 1.298 ms. The other three models have CPU INT8 only; building their engines needs another GPU session.",
    ),
    "Benchmark inference times across all formats": (
        "DONE",
        "models/optimization/benchmark.py; benchmarks/reports/BENCHMARKS.md",
        "fp32 and INT8 ONNX for all 4 models at batch 1 and 4, 100 runs per case interleaved across models, plus TensorRT fp32/fp16/INT8. INT8 (uint8, percentile calibration) on this CPU: 1.7x to 2.5x faster than fp32 for the three ResNets, 1.33x slower for yolov8n, 3.67x to 3.92x smaller.",
    ),
    "Comprehensive model validation pipeline": (
        "DONE",
        "models/validation/validate.py; benchmarks/reports/validation.json",
        "Artifacts, determinism, batch invariance, output sanity, robustness, inference errors, accuracy, calibration and latency. All 4 models pass; the fine-tuned model scores 78.91% top-1 on 10,000 images.",
    ),
    "A/B testing framework": (
        "DONE",
        "models/validation/ab_test.py; benchmarks/reports/ab_test.json",
        "Paired McNemar test with a confidence interval, latency comparison, hash-based traffic splitting. name:version@runtime compares runtimes of one model. Run on all 10,000 validation images, fp32 against INT8: 78.91% against 78.38%, p = 0.002, keep fp32.",
    ),
    "Model drift detection": (
        "DONE",
        "models/validation/drift.py; benchmarks/reports/drift_report.json; benchmarks/reports/drift_report_shift.json",
        "KS test, chi-square and PSI, requiring significance and an effect size. Run on real images: a Tiny-ImageNet control shows no drift in any of 12 tests; against COCO photos, 8 of 12 tests drift, severity high, confidence 0.668 to 0.373 (PSI 1.51).",
    ),
    "Performance regression testing": (
        "DONE",
        "models/validation/regression.py; benchmarks/baselines.json; benchmarks/reports/regression.json",
        "Baselines with per-metric tolerances and hardware fingerprinting. Baselines for all 4 models at 100 iterations. Accuracy gates pass; latency gates are documented as needing dedicated hardware, because this laptop varies 30% within minutes (evidence in docs/TECHNICAL.md section 7).",
    ),
    "models/ directory with training scripts": (
        "DONE",
        "models/",
        "training/, optimization/, validation/, cards/, artifacts/, registry.py",
    ),
    "benchmarks/ directory with performance comparison": (
        "DONE",
        "benchmarks/reports/",
        "BENCHMARKS.md, benchmark_results.json, quantization.json, validation.json",
    ),
    # --- Part 3: testing ---------------------------------------------------
    "Unit tests, target >90% coverage": (
        "DONE",
        "tests/unit/",
        "95.8% on api/ and 94.6% on worker/, the whole request and batch path; CI gates api/ at 92%. External services are faked in unit tests (fakeredis, in-memory SQLite, a fake model runtime).",
    ),
    "Test model inference functions": (
        "DONE",
        "tests/unit/test_inference_service.py",
        "softmax, NMS, L2-norm, all 3 tasks.",
    ),
    "Test image preprocessing utilities": (
        "DONE",
        "tests/unit/test_image_processing.py; tests/unit/test_preprocessing_parity.py",
        "EXIF orientation, alpha compositing, letterbox round-trip, and training/serving preprocessing parity.",
    ),
    "Test API route handlers": (
        "DONE",
        "tests/unit/test_api_routes.py; tests/unit/test_batch.py",
        "Every endpoint, including batch ownership and error envelopes.",
    ),
    "Test service layer functions": (
        "DONE",
        "tests/unit/",
        "model, cache, inference and similarity-index services.",
    ),
    "Mock external dependencies": (
        "DONE",
        "tests/conftest.py",
        "fakeredis, aiosqlite and a FakeRuntime, so the suite needs no running services.",
    ),
    "Test database operations": (
        "DONE",
        "tests/integration/test_database.py; tests/integration/test_real_services.py",
        "Real SQL against SQLite, and against PostgreSQL and pgvector when reachable (verified).",
    ),
    "Test model loading and inference": (
        "DONE",
        "tests/integration/test_model_loading.py; tests/integration/test_quantized_fidelity.py",
        "Real ONNX artifacts, including every INT8 model against its fp32 twin.",
    ),
    "Test API endpoint integration": (
        "DONE",
        "tests/integration/test_api_endpoints.py; tests/e2e/; scripts/smoke_test_api.py",
        "In-process integration tests plus end-to-end tests through nginx against the running stack, all passing live. smoke_test_api.py checks every endpoint of any deployment and exits non-zero on failure.",
    ),
    "Stress testing for model inference": (
        "DONE",
        "tests/performance/test_performance.py, tests/performance/locustfile.py",
        "Concurrency, sustained load and throughput; locust for HTTP-level load.",
    ),
    "Memory usage profiling": (
        "DONE",
        "tests/performance/test_performance.py; benchmarks/reports/performance_tests.txt",
        "Resident memory measured across repeated inference, batches and preprocessing; growth levels off.",
    ),
    "Pytest configuration with fixtures": ("DONE", "pytest.ini, tests/conftest.py", ""),
    "Test database setup/teardown": ("DONE", "tests/conftest.py", "Fresh in-memory DB per test."),
    "Mock services for external dependencies": ("DONE", "tests/conftest.py", ""),
    "Continuous testing with GitHub Actions": ("DONE", ".github/workflows/ci.yml", ""),
    "tests/ directory with complete test suite": (
        "DONE",
        "tests/",
        "1,434 tests: 1,230 unit, 161 integration, 28 end-to-end, 15 performance, plus a Locust load test.",
    ),
    "Test configuration and fixtures": ("DONE", "tests/conftest.py", ""),
    "Performance test reports": ("DONE", "benchmarks/reports/", ""),
    "CI/CD pipeline configuration": (
        "DONE",
        ".github/workflows/",
        "ci.yml: ruff, black and mypy (all enforced), tests on 3.11 and 3.12 with a coverage gate, a strict INT8 fidelity check, security scan, image build and smoke test, end-to-end against the deployed stack, real model export. release.yml publishes scanned, attested images on a tag; drift-watch.yml runs the retraining decision weekly.",
    ),
    # --- Part 4: containerisation -----------------------------------------
    "Multi-stage builds": (
        "DONE",
        "Dockerfile, docker/Dockerfile.worker, docker/Dockerfile.gpu",
        "Builder stage keeps compilers out of the runtime image. A third "
        "Dockerfile builds the CUDA/TensorRT serving image.",
    ),
    "Security best practices (non-root": (
        "DONE",
        "docker/",
        "Non-root uid 10001, slim base, no-new-privileges, read-only rootfs in production.",
    ),
    "Optimised for layer caching": (
        "DONE",
        "docker/",
        "Dependencies installed before application code is copied.",
    ),
    "Separate images for different services": (
        "DONE",
        "docker/",
        "api, worker and gateway built separately.",
    ),
    "Service: api-gateway": (
        "DONE",
        "docker/nginx/",
        "Nginx: least_conn balancing, edge rate limits, /metrics restricted to internal "
        "ranges. VERIFIED healthy.",
    ),
    "Service: ml-api": (
        "DONE",
        "docker-compose.yml",
        "VERIFIED healthy; served real predictions through the gateway.",
    ),
    "Service: worker": (
        "DONE",
        "docker/Dockerfile.worker; worker/",
        "Healthy in dev and production; batch jobs verified end to end, with job state written to Postgres and metrics served to Prometheus.",
    ),
    "Service: redis": (
        "DONE",
        "docker-compose.yml",
        "VERIFIED healthy; cache hits confirmed live.",
    ),
    "Service: postgres": ("DONE", "docker-compose.yml", "VERIFIED healthy."),
    "Service: prometheus": (
        "DONE",
        "monitoring/prometheus/",
        "Scrapes every API replica and worker; 11 alert rules validated by promtool.",
    ),
    "Service: grafana": (
        "DONE",
        "monitoring/grafana/",
        "Data source and 18-panel dashboard auto-provisioned.",
    ),
    "Health checks for all services": (
        "DONE",
        "docker-compose.yml",
        "All 7 dev services and all 10 production containers report healthy, verified live. Liveness checks no dependencies.",
    ),
    "Proper resource limits": (
        "DONE",
        "docker-compose.yml + docker-compose.prod.yml",
        "CPU and memory limits plus reservations per service.",
    ),
    "Volume management": (
        "DONE",
        "docker-compose.yml",
        "Named volumes; model artifacts mounted read-only.",
    ),
    "Network configuration and service discovery": (
        "DONE",
        "docker-compose.yml",
        "Three segmented networks; DNS-based discovery, no hardcoded IPs.",
    ),
    "Dockerfile for each service": ("DONE", "docker/", ""),
    "docker-compose.yml for local development": (
        "DONE",
        "docker-compose.yml",
        "All 7 services healthy, verified live.",
    ),
    "docker-compose.prod.yml for production": (
        "DONE",
        "docker-compose.prod.yml",
        "Replicas, no exposed ports, read-only rootfs, secrets required (verified it "
        "refuses to start without them).",
    ),
    # --- Part 4 ------------------------------------------------------------
    # --- Part 1 / 2 deliverables ------------------------------------------
    "Comprehensive model cards": (
        "DONE",
        "models/cards/",
        "One card per model: architecture rationale, measured latency, validation "
        "results, and an explicit limitations section (bias, calibration, scaling, "
        "licensing).",
    ),
    "Complete FastAPI application with all endpoints": (
        "DONE",
        "api/",
        "19 paths in the OpenAPI spec, verified live through the Docker stack.",
    ),
    "Comprehensive API documentation": (
        "DONE",
        "docs/API.md + /docs + /redoc",
        "Auto-generated OpenAPI enriched with auth, rate limits and the error "
        "envelope, plus a hand-written reference.",
    ),
    "Postman collection or OpenAPI spec": (
        "DONE",
        "docs/openapi.json",
        "OpenAPI 3.1, generated from the app, 19 paths. Importable into Postman.",
    ),
    "Documentation for deployment and scaling": (
        "DONE",
        "docs/DEPLOYMENT.md, k8s/README.md",
        "First run, production checklist, scaling guide with measured sizing, a "
        "Kubernetes section covering when to move, prerequisites, what changes "
        "from Compose and how to verify it, then monitoring, model "
        "rollout/rollback, troubleshooting, backup and recovery. The manifests "
        "have their own README with the cluster evidence.",
    ),
    # --- Documentation -----------------------------------------------------
    "README: Architecture overview and design decisions": (
        "DONE",
        "README.md + docs/TECHNICAL.md",
        "A diagram per deployment target, Compose and Kubernetes, with the "
        "middleware ordering rationale and layering. TECHNICAL.md §6 covers why "
        "the cluster topology differs: HPA instead of a typed replica count, "
        "ingress instead of the gateway container, fetched rather than baked-in "
        "artifacts, and migrations in an init container.",
    ),
    "README: Setup and installation instructions": (
        "DONE",
        "README.md",
        "Quick start: clone, git lfs pull, copy .env, docker compose up.",
    ),
    "README: API usage examples": (
        "DONE",
        "README.md, docs/API.md",
        "curl examples with real responses.",
    ),
    "README: Performance benchmarks": (
        "DONE",
        "README.md, benchmarks/reports/BENCHMARKS.md",
        "Measured on named hardware, reproducible with one command.",
    ),
    "README: Known limitations and future improvements": (
        "DONE",
        "README.md",
        "7 limitations stated plainly, and next steps in priority order.",
    ),
    "API docs: Complete endpoint documentation": (
        "DONE",
        "docs/API.md",
        "Every endpoint, all parameters.",
    ),
    "API docs: Request/response examples": (
        "DONE",
        "docs/API.md",
        "Real captured responses, not invented ones.",
    ),
    "API docs: Error handling guide": (
        "DONE",
        "docs/API.md",
        "All 14 error codes with retry guidance.",
    ),
    "API docs: Authentication guide": (
        "DONE",
        "docs/API.md",
        "API key and JWT, with failure modes.",
    ),
    "Technical doc: Model selection and optimisation rationale": (
        "DONE",
        "docs/TECHNICAL.md",
        "Why each model, what was traded away, and the INT8 finding in full.",
    ),
    "Technical doc: Performance benchmarking results": (
        "DONE",
        "docs/TECHNICAL.md",
        "Latency, batching, load test, verified system properties.",
    ),
    "Technical doc: System architecture decisions": (
        "DONE",
        "docs/TECHNICAL.md",
        "7 decisions with reasoning, including the three different failure policies.",
    ),
    "Technical doc: Scalability considerations": (
        "DONE",
        "docs/TECHNICAL.md",
        "What scales, what does not (the similarity index), sizing from measurement.",
    ),
    "Documented assumptions": (
        "DONE",
        "docs/ASSUMPTIONS.md; docs/AUDIT.md",
        "How the brief was read, fixes to 3 provided scripts, and a review against the brief with every issue found and fixed.",
    ),
    # --- Quality standards -------------------------------------------------
    "PEP 8 compliance, type hints, docstrings": (
        "DONE",
        "pyproject.toml",
        "ruff, black and mypy all clean and enforced in CI. Each disabled lint rule carries a written reason.",
    ),
    "Test coverage minimum 85%": (
        "DONE",
        "pytest --cov",
        "95.8% on api/ and 94.6% on worker/. models/ is 77.2% because it holds the GPU training loop and CLIs that no request touches.",
    ),
    "Performance: sub-second inference": (
        "DONE",
        "benchmarks/reports/BENCHMARKS.md",
        "p99 at batch 1: resnet50 211.7 ms, resnet50-tiny-imagenet 277.1 ms, yolov8n 307.5 ms, resnet50-embed 357.3 ms. Every model inside 1 s in both precisions; the slowest single-image p99 is resnet50-embed fp32 at 357.3 ms.",
    ),
    "Security: no hardcoded secrets": (
        "DONE",
        "api/config.py, api/utils/validators.py",
        "Fails closed on auth; production refuses to start without secrets "
        "(verified). SSRF guard, decompression-bomb guard, magic-byte format "
        "detection, TorchScript not pickle, JWT algorithm pinning.",
    ),
    "Scalability: design for horizontal scaling": (
        "DONE",
        "k8s/; docs/TECHNICAL.md; docker-compose.prod.yml",
        "Stateless API, Redis-backed rate limiting across replicas, independent worker scaling, pgvector similarity index in production and Kubernetes. Production overlay verified live with 3 API replicas and 2 workers; HPA verified on a kind cluster.",
    ),
    "Environment-based configuration": (
        "DONE",
        "api/config.py, .env.example",
        "Pydantic Settings; refuses to start insecure in staging/production.",
    ),
}


def apply_updates() -> list[Item]:
    """Return ITEMS with UPDATES applied."""
    out: list[Item] = []
    unmatched = set(UPDATES)
    for item in ITEMS:
        for key, (status, evidence, notes) in UPDATES.items():
            if key.lower() in item.requirement.lower():
                unmatched.discard(key)
                item = item._replace(
                    status=status,
                    evidence=evidence or item.evidence,
                    notes=notes or item.notes,
                )
                break
        out.append(item)
    if unmatched:
        # A key that matches nothing means the requirement text was edited and
        # the update is silently doing nothing - surface it loudly instead.
        raise ValueError(f"UPDATES keys matched no requirement: {sorted(unmatched)}")
    return out


__all__ = ["ITEMS", "UPDATES", "Item", "apply_updates"]
