# Multi-Model Computer Vision API

A production MLOps system serving **image classification**, **object
detection** and **image similarity search** behind one API — with
containerisation, per-tier rate limiting, background batch processing,
monitoring, drift detection and a 370-test suite.

Built for the Applied Computing ML Engineer challenge. The original brief is
preserved at [`docs/CHALLENGE.md`](docs/CHALLENGE.md).

---

## Status

| | |
| --- | --- |
| **CI** | All 7 jobs green on Python 3.11 and 3.12 |
| **Tests** | 404 passing locally; 390 passing / 14 skipped on CI |
| **Coverage** | 83.9% overall; 85-100% on critical paths |
| **Lint** | `ruff` and `black` clean |
| **Stack** | 7 services, all verified healthy |
| **Classifier** | 77.66% top-1 on Tiny-ImageNet (200 classes), validated 8/8 |
| **Latency** | 0.73 ms p50 on A100 via TensorRT fp16; 43-121 ms on CPU |
| **Load tested** | 1,677 requests, 0.2% failures, p95 320 ms, 38.7 req/s |

The 14 CI skips are the Tiny-ImageNet dataset tests. The dataset is 519 MB
across 120,203 files and is not committed — CI downloads and caches it, and
that step is deliberately non-fatal so an external host being down cannot turn
the build red.

Progress against every line of the brief is tracked in
[`DELIVERABLES_CHECKLIST.xlsx`](DELIVERABLES_CHECKLIST.xlsx), regenerated from
`scripts/checklist_data.py`.

---

## Quick start

Assumes **Git**, **Python 3.11 or 3.12** and **Docker Desktop** are installed.
Git LFS is needed too, because the model files are stored there:

```bash
git lfs install          # once per machine; see https://git-lfs.com
```

### 1. Get the code

```bash
git clone https://github.com/pankaj29/ML-Engineer-Challenge.git
cd ML-Engineer-Challenge
git lfs pull             # fetches the model files (~460 MB)
```

Without `git lfs pull` the `.onnx` files are 130-byte placeholders and the API
will fail to load a model.

### 2. Create and activate a virtual environment

**Do not skip this.** Installing into your system Python will fight with
whatever else is already there.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. If PowerShell refuses with a
script-execution error, allow it for this session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Configuration

```bash
cp .env.example .env     # PowerShell: copy .env.example .env
```

### 4. Export the models (one-off, ~5 minutes)

```bash
python -m pip install --upgrade pip
pip install -r requirements-train.txt onnxscript
python scripts/prepare_models.py
```

### 5. Start the stack

```bash
docker compose up -d
docker compose ps        # every service should reach "healthy"
```

### 6. Confirm

```bash
# bash - macOS / Linux. Will NOT work in PowerShell.
curl http://localhost/api/v1/health
```

Expect `{"status": "healthy", ...}`. If you get `SERVICE_UNAVAILABLE`, see
*If something goes wrong* below.

---

## Using the API

One worked example here; the complete reference for all 18 endpoints - both
shells, every parameter, every response shape - is in
[`docs/API.md`](docs/API.md).

> **On Windows**, the `curl` examples in this repo are bash and will not run in
> PowerShell: `curl` is an alias for `Invoke-WebRequest`, `\` is not a line
> continuation, and there is no `base64` command. Use the PowerShell block
> below, or the helpers in [`docs/API.md`](docs/API.md#before-you-start-bash-or-powershell)
> which reduce every call to one line.

### Classify an image

```bash
# bash - macOS / Linux. Will NOT work in PowerShell.
curl -X POST http://localhost/api/v1/classify \
     -H "X-API-Key: dev-key-pro" \
     -H "Content-Type: application/json" \
     -d "{\"image_base64\": \"$(base64 -w0 photo.jpg)\", \"top_k\": 5}"
```

```powershell
# Windows PowerShell
$bytes = [IO.File]::ReadAllBytes((Resolve-Path "photo.jpg").Path)
$body  = @{ image_base64 = [Convert]::ToBase64String($bytes); top_k = 5 } | ConvertTo-Json

$r = Invoke-RestMethod -Uri "http://localhost/api/v1/classify" -Method Post `
        -Headers @{ "X-API-Key" = "dev-key-pro" } `
        -ContentType "application/json" -Body $body

$r.predictions | Format-Table rank, label, confidence -AutoSize
```

```json
{
  "predictions": [
    {"class_id": 101, "label": "tusker",           "confidence": 0.309, "rank": 1},
    {"class_id": 386, "label": "African elephant", "confidence": 0.243, "rank": 2}
  ],
  "top_prediction": {"class_id": 101, "label": "tusker", "rank": 1},
  "model":  {"name": "resnet50", "version": "1.0.0", "runtime": "onnx"},
  "timing": {"preprocess_ms": 3.1, "inference_ms": 11.4, "total_ms": 15.2},
  "correlation_id": "0f3c...", "cached": false
}
```

Note what comes back besides the answer: which model version produced it, a
per-stage timing breakdown, a `correlation_id` for finding this exact request
in the logs, and whether it was served from cache.

### The rest of the API

| | |
| --- | --- |
| **Object detection** | `POST /api/v1/detect` - bounding boxes, in pixels of the original image |
| **Image similarity** | `POST /api/v1/similarity/{embed,index,search}` - 2048-dim vectors and nearest-neighbour search |
| **Batch** | `POST /api/v1/batch` - background jobs for many images, returns a job id to poll |
| **File upload** | Add `/upload` to any inference endpoint for multipart instead of base64 |
| **Models** | `GET /api/v1/models` - what is registered and which is default |
| **Health** | `/health`, `/health/live`, `/health/ready` - full, liveness, readiness |

Full reference: **[`docs/API.md`](docs/API.md)**. Interactive docs, generated
from the code: **<http://localhost:8000/docs>**.

### If something goes wrong

| Symptom | Cause and fix |
| --- | --- |
| `pip` reports conflicts with packages you have never heard of (`librosa`, `transformers`, `mcp`) | You are installing into your system Python. Go back to step 2 and activate the virtual environment. Those warnings are about *other* projects on your machine, not this one. |
| `SERVICE_UNAVAILABLE` from `http://localhost/...` but `http://localhost:8000/...` works | The gateway cached the API's old IP address. `docker compose restart api-gateway`. Fixed in the nginx config, so this should only affect stacks started before that change. |
| `ModelLoadError` / `503` on every request | The model files are LFS placeholders. Run `git lfs pull`. |
| `docker compose ps` shows a service as `unhealthy` | `docker compose logs <service> --tail 50`. The API needs up to 90 seconds on first start while it loads three models. |
| Port 80 already in use | `GATEWAY_PORT=8080 docker compose up -d`, then use `http://localhost:8080`. |
| PowerShell: *"The term 'base64' is not recognized"* | `base64` is a Unix tool. Use `[Convert]::ToBase64String([IO.File]::ReadAllBytes("photo.jpg"))`. |
| PowerShell: *"Could not find file"* naming the **wrong folder** | `[IO.File]` resolves relative paths against .NET's current directory, which `cd` does not change. Wrap the path: `(Resolve-Path "photo.jpg").Path`. |
| PowerShell: *"Cannot bind parameter 'Headers'"* | In PowerShell, `curl` is an **alias for `Invoke-WebRequest`**, which takes a dictionary, not `-H` strings. Use `curl.exe` for real curl, or `Invoke-RestMethod` with `-Headers @{...}` as shown above. |

Interactive docs: **<http://localhost:8000/docs>**

---

## Documentation

| Document | What it covers |
| --- | --- |
| [**API.md**](docs/API.md) | Every endpoint, request/response examples, errors, authentication |
| [**TECHNICAL.md**](docs/TECHNICAL.md) | Model selection, optimisation results, architecture, scalability |
| [**ASSUMPTIONS.md**](docs/ASSUMPTIONS.md) | Decisions, gaps, and every bug found along the way |
| [**DEPLOYMENT.md**](docs/DEPLOYMENT.md) | Production deployment, scaling, troubleshooting |
| [**BENCHMARKS.md**](benchmarks/reports/BENCHMARKS.md) | Full latency numbers across formats |
| [**Model cards**](models/cards/) | What each model does, how well, and where it fails |
| [`docs/openapi.json`](docs/openapi.json) | OpenAPI 3.1 spec — import into Postman |

---

## Architecture

```text
                    ┌──────────────┐
   client ─────────▶│  api-gateway │  Nginx: load balancing, edge rate
                    └──────┬───────┘  limiting, body caps, internal /metrics
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │    ml-api     │  × N          │    ml-api     │  FastAPI
   │  Monitoring   │               │               │  · auth + per-tier limits
   │  ↓ CORS       │               │               │  · validation
   │  ↓ Auth       │               │               │  · ONNX inference
   │  ↓ RateLimit  │               │               │  · concurrency cap
   │  ↓ routes     │               │               │
   └───┬───────┬───┘               └───────────────┘
       │       │
       ▼       ▼
┌────────────┐  ┌────────────┐          ┌──────────────┐
│   redis    │  │  postgres  │          │    worker    │ × M
│ cache      │  │ inference  │◀─────────│ Celery       │
│ broker     │─▶│ log + jobs │          │ batch jobs   │
│ limiter    │  └────────────┘          └──────────────┘
└─────┬──────┘
      │ scrape
┌─────┴────────┐   ┌────────────┐
│  prometheus  │──▶│  grafana   │  22-panel dashboard, 11 alert rules
└──────────────┘   └────────────┘
```

**Middleware order is deliberate.** Monitoring is outermost so it assigns the
correlation ID before anything else runs and times *every* request — including
ones rejected by authentication. Auth precedes rate limiting because the limit
depends on the caller's tier. Rate limiting is innermost so a rejected request
never touches a model.

Full reasoning in [TECHNICAL.md](docs/TECHNICAL.md).

---

## The three models

| Task | Model | p50 (CPU) | Size | Endpoint |
| --- | --- | ---: | ---: | --- |
| Classification | ResNet-50 (ImageNet-1k) | 84.8 ms | 97.4 MB | `POST /api/v1/classify` |
| Detection | YOLOv8n (COCO) | 120.6 ms | 12.1 MB | `POST /api/v1/detect` |
| Similarity | ResNet-50 embeddings | 43.4 ms | 89.6 MB | `POST /api/v1/similarity/*` |

Each has a [model card](models/cards/) documenting its measured performance
and — more importantly — its limitations.

> The brief's overview names three tasks while its numbered list gives two.
> The third model is similarity search, per the overview; confirmed before
> building. See [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) §1.1.

---

## The most interesting result

The brief asks for INT8 quantization. It is applied to all three models — and
measurement showed the obvious approach makes the system **dramatically
worse**:

| ResNet-50, batch 1 | p50 latency | Size |
| --- | ---: | ---: |
| ONNX float32 | **75.7 ms** | 97.4 MB |
| INT8 **dynamic** | **1008.0 ms** | 24.5 MB |
| INT8 **static QDQ** | **104.6 ms** | 24.9 MB |

Dynamic quantization was **13x slower than float32**. It recomputes activation
scales on every call and falls back to poorly-optimised integer convolution
kernels — fine for a transformer, catastrophic for a convolutional network.

Switching to **static QDQ** quantization, calibrated on 100 real images, made
it ~10x faster than dynamic. Even so it remains ~1.4x slower than float32 on
this CPU, while being 3.9x smaller.

**So float32 is the serving default**, with INT8 registered alongside and
selectable per request. Shipping a 13x-slower "optimisation" as the default,
because the brief said to apply quantization, would have been the wrong call.

Full analysis in [TECHNICAL.md §2](docs/TECHNICAL.md#2-optimisation-what-worked-and-what-did-not).

---

## The fine-tuned classifier

ResNet-50 on Tiny-ImageNet — 200 classes, all 100,000 training images, 60
epochs on an NVIDIA A100-SXM4-40GB. **77.66% top-1, 91.52% top-5** against a
0.5% random baseline, in 33.9 minutes.

![Training curves: loss, validation accuracy, and the warmup + cosine learning rate schedule](docs/images/training-curves.png)

Produced by the training run itself, not redrawn — the three panels are loss,
validation accuracy, and the learning-rate schedule, showing all three
techniques the brief asks for: mixed precision (fp16 AMP + `GradScaler`),
gradient clipping (`clip_grad_norm_` at 1.0) and LR scheduling (5% linear
warmup into cosine decay).

**The honest reading of the middle panel:** top-1 reaches 76.5% by epoch 2 and
the remaining 58 epochs add about a point. Transfer learning from ImageNet-1k
does nearly all the work immediately; the long cosine tail is worth ~1.1
points. Around 25-30 epochs would have captured most of it.

Full detail, including two counter-intuitive findings about resolution and
learning rate, is in
[`models/cards/resnet50-tiny-imagenet.md`](models/cards/resnet50-tiny-imagenet.md).

### Against TensorRT on the same GPU

| Runtime | Precision | p50 | Throughput | Size |
| --- | --- | ---: | ---: | ---: |
| TensorRT | fp16 | **0.729 ms** | **1369 img/s** | 46.0 MB |
| TensorRT | fp32 | 1.059 ms | 972 img/s | 91.5 MB |

---

## Measured performance

Intel Core Ultra 7 155H, 22 logical cores, **CPU only**. ONNX Runtime 1.26.0.

### Single image — the requirement is sub-second

| Model | p50 | p95 | p99 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| resnet50 | 84.8 ms | 109.1 ms | 131.5 ms | 13.1/s |
| yolov8n | 120.6 ms | 153.5 ms | 235.2 ms | 8.0/s |
| resnet50-embed | 43.4 ms | 278.3 ms | 387.9 ms | 10.8/s |

**All three meet the requirement at p99**, the slowest ~4x inside budget.

### End-to-end, through the full Docker stack

20 concurrent users, 45 seconds, mixed workload:

| Metric | Result |
| --- | --- |
| Requests | 1,677 |
| Failures | 4 (0.2%) |
| p50 / p95 / p99 | 90 / 320 / 1,300 ms |
| Throughput | 38.7 req/s |

### Verified system properties

| Property | Evidence |
| --- | --- |
| Concurrency ceiling honoured | Peak in-flight 4 against a limit of 4 |
| No memory leak | Growth decelerates across 100 inferences |
| No degradation under load | p50 8.3 ms → 7.0 ms over a sustained run |
| Invalid input is cheap | 0.019 ms to reject a malformed image |
| One bad image cannot fail a batch | Live run: 3 succeeded, 1 failed in isolation |

Reproduce: `python -m models.optimization.benchmark`

---

## Features

### Part 1 — Models and optimisation

* Three models covering classification, detection and similarity
* Tiny-ImageNet fine-tuning pipeline with **mixed precision** (AMP + GradScaler
  on CUDA, bf16 on CPU), **gradient clipping** and **cosine LR scheduling with
  warmup**
* Custom augmentation written from scratch: RandAugment (13 operations),
  RandomResizedCrop, RandomErasing, MixUp, CutMix
* ONNX export with **numerical verification** against PyTorch (max diff < 4e-06)
* INT8 quantization, static and dynamic, with measured accuracy cost
* TensorRT export — **executed on an A100 (TensorRT 11.3)**: fp16 at 0.729 ms
  p50 / 1369 img/s, 1.45x faster and half the size of fp32
* Validation pipeline: determinism, batch invariance, output sanity,
  robustness, calibration (ECE), latency
* A/B testing with a **paired McNemar test** and confidence intervals
* Drift detection: KS test, chi-square, PSI — requiring *both* statistical
  significance and a meaningful effect size
* Performance regression testing with hardware fingerprinting

### Part 2 — Production API

* All six required endpoints, plus similarity and batch management
* Per-tier rate limiting via a **Redis Lua token bucket** (atomic across
  replicas), with a per-process fallback
* Comprehensive image validation: size, format from magic bytes,
  decompression-bomb guard, **SSRF protection** on `image_url`
* Async batch processing through Celery
* Model versioning with per-request pinning and hot reload
* Graceful degradation with an explicit `degraded` flag
* Structured JSON logging with **correlation IDs** end to end
* One error envelope for every failure

### Part 3 — Testing

* 370 tests: 286 unit, 67 integration, 15 performance, 2 load-test classes
* Runs with **no external services** — fakeredis, in-memory SQLite, fake runtimes
* Real-artifact integration tests that skip cleanly when artifacts are absent
  **or are unfetched Git LFS pointers**, naming the remedy in the skip message
* Memory-leak profiling and concurrency verification
* Locust load testing against the live stack
* CI with lint, type-check, test, coverage gate, Docker build and security scan

### Part 4 — Containerisation

* Multi-stage builds, non-root user (uid 10001), slim base images
* Seven services, all with health checks, **all verified healthy**
* Production overlay: replicas, no exposed ports, read-only root filesystems,
  rolling updates, and **secrets that are required, not defaulted**
* Three segmented networks; DNS service discovery, no hardcoded IPs
* Prometheus with 11 alert rules; Grafana auto-provisioned with a 22-panel
  dashboard

---

## Project layout

```text
├── api/                      FastAPI application
│   ├── main.py               app factory, lifespan, middleware stack
│   ├── config.py             12-factor settings; fails fast if insecure
│   ├── exceptions.py         error hierarchy and handlers
│   ├── logging_config.py     structured JSON logging + correlation IDs
│   ├── dependencies.py       shared FastAPI dependencies
│   ├── routers/              classification, detection, similarity, batch,
│   │                         models, health, metrics
│   ├── models/               request and response schemas
│   ├── services/             model, cache, inference, database, index
│   ├── middleware/           auth, rate_limit, monitoring
│   └── utils/                image_processing, validators
├── models/
│   ├── training/             train_classifier, augmentation, dataset
│   ├── optimization/         export_onnx, export_tensorrt, quantize, benchmark
│   ├── validation/           validate, ab_test, drift, regression
│   ├── cards/                one model card per model
│   └── registry.py           model registry CLI
│   └── artifacts/            .onnx / .pt model files -- tracked in Git LFS
├── worker/                   Celery app and batch tasks
├── db/                       SQLAlchemy models
├── tests/                    unit, integration, performance
├── notebooks/                colab_gpu_pipeline.ipynb (generated -- see below)
├── Dockerfile                main application container (the API)
├── docker/                   Dockerfile.worker, nginx/
├── monitoring/               prometheus config + alerts, grafana provisioning
├── scripts/                  prepare_models, download_datasets, checklist
├── benchmarks/               baselines and generated reports
├── docs/                     API, TECHNICAL, ASSUMPTIONS, DEPLOYMENT, openapi
├── .gitattributes            Git LFS rules for the model files
└── .github/workflows/        CI pipeline
```

Two paths are **generated**, not hand-edited:

* `notebooks/colab_gpu_pipeline.ipynb` comes from
  `scripts/build_colab_notebook.py`. Editing the notebook directly is
  overwritten on the next build, and CI fails if the two drift apart
  (`python scripts/build_colab_notebook.py --check`).
* `DELIVERABLES_CHECKLIST.xlsx` comes from `scripts/generate_checklist.py`.

`models/artifacts/` is tracked with **Git LFS** (see `.gitattributes`): those
files cannot be regenerated without a GPU session, unlike the dataset, which
is excluded because one command rebuilds it. A clone made without git-lfs gets
130-byte pointer files and the model tests skip with `git lfs pull` as the
stated remedy.

Two files extend the brief's prescribed structure: `api/config.py` (required
by "environment-based configuration" and "no hardcoded secrets") and
`api/dependencies.py` (so image extraction is defined once rather than per
router). Extra routers exist because the brief requires those endpoints.

---

## Development

Activate the virtual environment from step 2 of the Quick start first -
`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate`
elsewhere. Your prompt should read `(.venv)`.

```bash
pip install -r requirements-dev.txt

pytest tests/ -v                          # everything except performance
pytest tests/ --cov=api --cov-report=html # with a coverage report
pytest tests/performance -m performance -s # timing and memory
ruff check api/ models/ worker/ tests/    # lint
black api/ models/ worker/ tests/         # format
```

### Working with models

```bash
python scripts/prepare_models.py                   # export and register all three
python -m models.registry list                     # what is registered
python -m models.registry validate                 # do the artifacts exist?
python -m models.validation.validate               # full validation suite
python -m models.optimization.benchmark            # latency across formats
python -m models.validation.regression check --model resnet50:1.0.0
```

### Training

**Training always runs on the complete dataset** — all 200 classes, all
100,000 training and 10,000 validation images, every batch. There is no option
to subset it. A startup guard (`verify_full_dataset`) counts what is on disk,
compares it against what the dataloaders picked up, and refuses to train if
anything is missing.

```bash
# The standard run. Uses CUDA automatically when a GPU is available.
python -m models.training.train_classifier --epochs 30 --batch-size 256 --device auto

# Faster on CPU at some accuracy cost — still all 200 classes, all images.
python -m models.training.train_classifier --arch resnet18 --no-stem-adapt --epochs 10
```

Measured CPU cost on an Intel Core Ultra 7 155H (16 threads):

| Config | img/s | 1 epoch | 30 epochs |
| --- | ---: | ---: | ---: |
| resnet50 + 64px stem (default) | 3.8 | 7.6 h | 9.4 days |
| resnet50, original stem | 21.2 | 1.4 h | 1.7 days |
| resnet18 + 64px stem | 9.2 | 3.1 h | 3.9 days |
| resnet18, original stem | 75.0 | 23 min | 11.5 h |

The 64px stem adaptation dominates the cost: it is correct for 64px input, but
every layer after it then runs at 16x the spatial area. A GPU is roughly
30-60x faster, putting the default config well under an hour.

---

## Security

* **No hardcoded secrets.** Production refuses to start without them —
  verified in CI.
* **Fails closed on authentication.** No keys configured means every request
  is rejected; there is no default credential.
* **Constant-time key comparison**; keys never appear in logs, only a
  non-reversible fingerprint.
* **SSRF protection** on `image_url`: scheme and port allow-lists, redirects
  disabled, and every resolved address checked against private, loopback and
  link-local ranges.
* **Decompression-bomb guard**: headers are parsed and pixel counts checked
  *before* any pixel buffer is allocated.
* **Content-based format detection** from magic bytes, never from a filename
  or a client-declared content type.
* **TorchScript, not pickle** — loading a pickled checkpoint would execute
  arbitrary code from the artifact.
* **JWT algorithm pinning**, so an `alg: none` forgery is rejected.
* **Non-root containers**, read-only root filesystems in production,
  `no-new-privileges`.
* **No images stored** — only a SHA-256 hash.

---

## Known limitations

Stated plainly; the full list with reasoning is in
[ASSUMPTIONS.md](docs/ASSUMPTIONS.md).

1. **TensorRT INT8 is not built** — fp32 and fp16 both are. In TensorRT 11 an
   INT8 engine needs a QDQ graph; `quantize.py` produces one, but it has not
   been run, so `precision="int8"` refuses rather than silently building fp32
   and labelling it INT8.
2. **Accuracy figures for the ImageNet-1k and COCO models are cited, not
   re-measured** — that needs the ImageNet
   and COCO validation sets. Behavioural correctness *was* verified end to end.
3. **The similarity index is per-process**, so it does not survive horizontal
   scaling. Options are set out in TECHNICAL.md.
4. **Confidence calibration varies by model** — the fine-tuned Tiny-ImageNet
   classifier measures ECE 0.063, but the ImageNet-1k model measures 0.22. Use
   the ranking, not the absolute scores, unless you have measured otherwise.
5. **YOLOv8 is AGPL-3.0**, which has real implications for commercial use.
6. **The gateway round-robins rather than least-connections.** nginx caches an
   upstream's DNS answer at startup, so an `upstream` block pointed at a
   container that is later rebuilt keeps calling a dead IP - observed here as a
   gateway stuck on a stale address for 22 hours. The fix resolves per request
   via a variable, which cannot reference an upstream block, so `least_conn`
   and upstream keepalive were given up to get self-healing. Reasoning is in
   `docker/nginx/nginx.conf`.

### Next, in priority order

1. Build a TensorRT INT8 engine from the existing QDQ graph
2. Move the similarity index to a shared store (pgvector or FAISS)
3. Measure accuracy properly against the real ImageNet and COCO validation sets
4. Calibrate confidence with temperature scaling
5. Add OpenTelemetry tracing

6. Restore least-connections balancing at the gateway (needs nginx Plus, or a
   hook that restarts the gateway when the API is recreated)
7. Alert when the rate limiter is running on local buckets rather than Redis

*Done since the first draft: the full Tiny-ImageNet fine-tune (77.66% top-1),
the TensorRT fp32/fp16 path (0.729 ms p50), a green CI pipeline, and a fix for
rate limiting that had silently been per-process rather than shared across
replicas.*

---

## A note on how this was built

Several defects were found by the system's own checks rather than by review,
and they are documented rather than quietly fixed — the
[full list is in ASSUMPTIONS.md §4-5](docs/ASSUMPTIONS.md). A few worth
knowing about:

* **The provided starter dataloader mislabelled the entire validation set.**
  `ImageFolder` cannot read Tiny-ImageNet's validation layout and silently
  assigned label 0 to all 10,000 images. It does not crash — validation
  accuracy just reads a meaningless 0.5%. Fixed in place; now verified to
  produce 200 distinct labels with exactly 50 images each.
* **ONNX export silently broke batching.** torch 2.9's default exporter
  ignores `dynamic_axes`. Caught by the export script's own verification.
* **A lint suppression corrupted the Redis Lua script.** A `# noqa` placed
  after the opening triple-quote became the first line of the Lua source.
  Redis would have rejected it — but only with a real Redis, which the unit
  tests never use. The whole suite was blind to it.
* **Three Prometheus metrics were defined but never incremented**, so a
  dashboard panel and an alert would have been permanently blank. Found by
  querying Prometheus after a live run rather than by trusting the code.

The general principle applied throughout: **a claim is not done until there is
evidence for it.** Every number in this README came from a command that was
actually run, on the hardware described.
