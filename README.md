# Multi-Model Computer Vision API

One API serving image classification, object detection and image similarity
search, with containerisation, per-tier rate limiting, background batch
processing, monitoring and drift detection.

Built for the Applied Computing ML Engineer challenge. The original brief is
at [`docs/CHALLENGE.md`](docs/CHALLENGE.md).

---

## Status

| | |
| --- | --- |
| CI | 6 jobs green on Python 3.11 and 3.12 |
| Tests | 1,181: 995 unit, 143 integration, 28 end-to-end, 15 performance |
| Coverage | 95.8% on `api/`; cache, database and rate limiting at 100% |
| Lint | `ruff` and `black` clean, `mypy` clean |
| Stack | 7 services, all healthy |
| Classifier | 78.91% top-1 on Tiny-ImageNet, 9 of 9 validation checks pass |
| Latency | 0.86 ms p50 on A100 via TensorRT fp16; 43-121 ms on CPU |
| Load tested | 1,677 requests, 0.2% failures, p95 320 ms, 38.7 req/s |

CI skips the Tiny-ImageNet dataset tests. The dataset is 519 MB across 120,203
files and is not committed. CI downloads and caches it, and that step is
non-fatal so an external host being down cannot turn the build red.

Progress against the brief is tracked in
[`DELIVERABLES_CHECKLIST.xlsx`](DELIVERABLES_CHECKLIST.xlsx), generated from
`scripts/checklist_data.py`.

---

## Quick start

You need Git, Python 3.11 or 3.12, and Docker Desktop. Git LFS too, because
the model files live there:

```bash
git lfs install          # once per machine; see https://git-lfs.com
```

### 1. Get the code

```bash
git clone https://github.com/pankaj29/ML-Engineer-Challenge.git
cd ML-Engineer-Challenge
git lfs pull             # fetches the model files, about 460 MB
```

Without `git lfs pull` the `.onnx` files are 130-byte placeholders and the API
cannot load a model.

### 2. Create a virtual environment

Do not skip this. Installing into your system Python will fight with whatever
else is already there.

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

### 4. Export the models

One-off, about five minutes.

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

### 6. Check it works

```bash
# bash. Will not work in PowerShell.
curl http://localhost/api/v1/health
```

You should get `{"status": "healthy", ...}`. If you get `SERVICE_UNAVAILABLE`,
see [If something goes wrong](#if-something-goes-wrong).

---

## Using the API

One worked example below. The full reference for all 18 endpoints, in both
shells, is in [`docs/API.md`](docs/API.md).

> On Windows the `curl` examples here are bash and will not run in PowerShell:
> `curl` is an alias for `Invoke-WebRequest`, `\` is not a line continuation,
> and there is no `base64` command. Use the PowerShell block below, or the
> helpers in [`docs/API.md`](docs/API.md#before-you-start-bash-or-powershell)
> that reduce every call to one line.

### Classify an image

The repository ships three photographs in [`samples/`](samples/) so these
examples run as written. `samples/dog.jpg` classifies as a Labrador retriever
and detects a dog, so one image exercises both endpoints. Any JPEG or PNG of
your own works just as well.

The simplest form uploads the file directly:

```bash
# bash or PowerShell, using real curl. On Windows type curl.exe, because
# PowerShell's `curl` is an alias for Invoke-WebRequest.
curl -X POST http://localhost/api/v1/classify/upload \
     -H "X-API-Key: dev-key-pro" \
     -F "file=@samples/dog.jpg" -F "top_k=5"
```

To send base64 in a JSON body instead, pipe it through stdin. Putting the
base64 in the command line itself looks tidier and breaks: a 50 KB photo is
67 KB of base64, which is past the maximum argument length, so the shell
fails with "Argument list too long" before curl runs.

```bash
# bash. Will not work in PowerShell.
curl -X POST http://localhost/api/v1/classify \
     -H "X-API-Key: dev-key-pro" \
     -H "Content-Type: application/json" \
     -d @- <<EOF
{"image_base64": "$(base64 -w0 samples/dog.jpg)", "top_k": 5}
EOF
```

```powershell
# Windows PowerShell
$bytes = [IO.File]::ReadAllBytes((Resolve-Path "samples\dog.jpg").Path)
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

Besides the answer you get which model version produced it, a per-stage timing
breakdown, a `correlation_id` for finding this request in the logs, and
whether it came from cache.

### The rest of the API

| | |
| --- | --- |
| Object detection | `POST /api/v1/detect`, bounding boxes in pixels of the original image |
| Image similarity | `POST /api/v1/similarity/{embed,index,search}`, 2048-dim vectors and nearest-neighbour search |
| Batch | `POST /api/v1/batch`, background jobs returning an id to poll |
| File upload | Add `/upload` to any inference endpoint for multipart instead of base64 |
| Models | `GET /api/v1/models`, what is registered and which is default |
| Health | `/health`, `/health/live`, `/health/ready` |

Full reference: [`docs/API.md`](docs/API.md). Interactive docs generated from
the code: <http://localhost:8000/docs>.

### If something goes wrong

| Symptom | Cause and fix |
| --- | --- |
| `pip` reports conflicts with packages you have never heard of (`librosa`, `transformers`, `mcp`) | You are installing into your system Python. Go back to step 2 and activate the virtual environment. Those warnings are about other projects on your machine. |
| `SERVICE_UNAVAILABLE` from `http://localhost/...` but `http://localhost:8000/...` works | The gateway cached the API's old IP. `docker compose restart api-gateway`. The nginx config now resolves per request, so this only affects stacks started before that change. |
| `ModelLoadError` or `503` on every request | The model files are LFS placeholders. Run `git lfs pull`. |
| A service shows as `unhealthy` | `docker compose logs <service> --tail 50`. The API needs up to 90 seconds on first start while it loads three models. |
| Port 80 already in use | `GATEWAY_PORT=8080 docker compose up -d`, then use `http://localhost:8080`. |
| The container serves an old model after you replace a file | Docker Desktop on Windows does not always propagate a bind-mounted file that was replaced rather than edited. Compare `docker compose exec ml-api md5sum models/artifacts/<file>` against the host, and rebuild if they differ. |
| PowerShell: "The term 'base64' is not recognized" | `base64` is a Unix tool. Use `[Convert]::ToBase64String([IO.File]::ReadAllBytes("samples\dog.jpg"))`. |
| PowerShell: "Could not find file" naming the **wrong folder** | The file exists but `[IO.File]` resolves relative paths against .NET's current directory, which `cd` does not change. Wrap the path: `(Resolve-Path "samples\dog.jpg").Path`. |
| `Argument list too long` from curl | The base64 is too big for a command-line argument. Use the `/upload` endpoint, or pipe the JSON through stdin with `-d @-`, as shown in [Classify an image](#classify-an-image). |
| PowerShell: "Cannot bind parameter 'Headers'" | `curl` is an alias for `Invoke-WebRequest`, which takes a dictionary rather than `-H` strings. Use `curl.exe`, or `Invoke-RestMethod` with `-Headers @{...}` as above. |

---

## Documentation

| Document | Covers |
| --- | --- |
| [API.md](docs/API.md) | Every endpoint, request and response examples, errors, authentication |
| [TECHNICAL.md](docs/TECHNICAL.md) | Model selection, optimisation results, architecture, scalability |
| [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | Decisions, known limits, deviations from the brief's scaffolding |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment, scaling, troubleshooting |
| [BENCHMARKS.md](benchmarks/reports/BENCHMARKS.md) | Latency across formats |
| [Model cards](models/cards/) | What each model does, how well, and where it fails |
| [openapi.json](docs/openapi.json) | OpenAPI 3.1 spec, importable into Postman |

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

The middleware order is chosen, not incidental. Monitoring sits outermost so
it assigns the correlation ID first and times every request, including ones
rejected by authentication. Auth comes before rate limiting because the limit
depends on the caller's tier. Rate limiting is innermost, so a rejected
request never touches a model.

Reasoning in [TECHNICAL.md](docs/TECHNICAL.md).

---

## The models

Four models covering three tasks:

| Task | Model | p50 (CPU) | Size | Endpoint |
| --- | --- | ---: | ---: | --- |
| Classification | ResNet-50 (ImageNet-1k) | 84.8 ms | 97.4 MB | `POST /api/v1/classify` |
| Classification | ResNet-50 fine-tuned on Tiny-ImageNet | 14.8 ms | 95.6 MB | same, `model_name` pinned |
| Detection | YOLOv8n (COCO) | 120.6 ms | 12.1 MB | `POST /api/v1/detect` |
| Similarity | ResNet-50 embeddings | 43.4 ms | 89.6 MB | `POST /api/v1/similarity/*` |

Each has a [model card](models/cards/) with its measured performance and, more
usefully, its limitations.

> The brief's overview names three tasks while its numbered list gives two.
> The third model is similarity search, per the overview, confirmed before
> building. See [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) §1.1.

---

## Quantization: the result worth reading

The brief asks for INT8 quantization. It is applied to all four models, and
measurement showed the obvious approach makes things much worse:

| ResNet-50, batch 1 | p50 latency | Size |
| --- | ---: | ---: |
| ONNX float32 | 75.7 ms | 97.4 MB |
| INT8 dynamic | 1008.0 ms | 24.5 MB |
| INT8 static QDQ | 104.6 ms | 24.9 MB |

Dynamic quantization was 13× slower than float32. It recomputes activation
scales on every call and falls back to poorly optimised integer convolution
kernels, which is fine for a transformer and bad for a convolutional network.

Static QDQ, calibrated on 100 real images, is about ten times faster than
dynamic. It is still roughly 1.4× slower than float32 on this CPU, while being
3.9× smaller.

So float32 is the serving default, with INT8 registered alongside and
selectable per request. Shipping a 13× slower optimisation as the default,
because the brief said to apply quantization, would have been the wrong call.

Full analysis in [TECHNICAL.md §2](docs/TECHNICAL.md#2-optimisation-what-worked-and-what-did-not).

---

## The fine-tuned classifier

ResNet-50 on Tiny-ImageNet: 200 classes, all 100,000 training images, 60
epochs at 224×224 on an A100-SXM4-40GB. **78.91% top-1, 92.12% top-5** against
a 0.5% random baseline, in 1.6 hours.

![Loss, validation accuracy, the learning-rate schedule, and raw versus EMA weights](docs/images/training-curves.png)

Plotted from the run's own history file by `scripts/plot_training_curves.py`.
The four panels show loss, validation accuracy, the learning-rate schedule and
the effect of weight averaging, and between them they cover all three
techniques the brief asks for: mixed precision (fp16 AMP with `GradScaler`),
gradient clipping (`clip_grad_norm_` at 1.0) and LR scheduling (5% linear
warmup into cosine decay).

Reading the accuracy panel: transfer learning from ImageNet-1k reaches 75.1%
by epoch 2, and the remaining 58 epochs add 3.8 points. About half of that
arrives after epoch 40, when the cosine anneal drops the learning rate by two
orders of magnitude. The long schedule earns its place, which was not obvious
in advance.

The rightmost panel is weight averaging. EMA beat the live weights in 53 of 60
epochs, and the shipped checkpoint is an EMA one.

More detail, including the three input resolutions measured and why the native
64×64 is the worst of them, is in
[`models/cards/resnet50-tiny-imagenet.md`](models/cards/resnet50-tiny-imagenet.md).

### TensorRT on the same GPU

| Runtime | Precision | p50 | Throughput | Size |
| --- | --- | ---: | ---: | ---: |
| TensorRT | fp16 | **0.859 ms** | **1158 img/s** | 46.0 MB |
| TensorRT | fp32 | 1.099 ms | 907 img/s | 91.5 MB |

---

## Measured performance

Intel Core Ultra 7 155H, 22 logical cores, CPU only, ONNX Runtime 1.26.0.

### Single image

The requirement is sub-second.

| Model | p50 | p95 | p99 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| resnet50 | 84.8 ms | 109.1 ms | 131.5 ms | 13.1/s |
| yolov8n | 120.6 ms | 153.5 ms | 235.2 ms | 8.0/s |
| resnet50-embed | 43.4 ms | 278.3 ms | 387.9 ms | 10.8/s |

All three meet it at p99, the slowest about 4× inside budget.

### End to end through the Docker stack

20 concurrent users, 45 seconds, mixed workload:

| Metric | Result |
| --- | --- |
| Requests | 1,677 |
| Failures | 4 (0.2%) |
| p50 / p95 / p99 | 90 / 320 / 1,300 ms |
| Throughput | 38.7 req/s |

### System properties checked

| Property | Evidence |
| --- | --- |
| Concurrency ceiling honoured | Peak in-flight 4 against a limit of 4 |
| No memory leak | Growth decelerates across 100 inferences |
| No degradation under load | p50 8.3 ms to 7.0 ms over a sustained run |
| Invalid input is cheap to reject | 0.019 ms |
| One bad image cannot fail a batch | 3 succeeded, 1 failed in isolation |

Reproduce with `python -m models.optimization.benchmark`.

---

## What is built

### Part 1, models and optimisation

- Four models covering classification, detection and similarity
- Tiny-ImageNet fine-tuning with mixed precision (AMP plus GradScaler on CUDA,
  bf16 on CPU), gradient clipping and cosine LR scheduling with warmup
- Augmentation written from scratch: RandAugment (13 operations),
  RandomResizedCrop, RandomErasing, MixUp, CutMix
- ONNX export verified numerically against PyTorch, max diff 3.46e-06
- INT8 quantization, static and dynamic, with the accuracy cost measured
- TensorRT built and benchmarked on an A100: fp16 at 0.859 ms p50 and
  1158 img/s, 1.28× faster and half the size of fp32
- Validation pipeline: determinism, batch invariance, output sanity,
  robustness, calibration, latency
- A/B testing with a paired McNemar test and confidence intervals
- Drift detection with KS test, chi-square and PSI, requiring both statistical
  significance and a meaningful effect size
- Performance regression testing with hardware fingerprinting

### Part 2, production API

- All six required endpoints, plus similarity and batch management
- Per-tier rate limiting through a Redis Lua token bucket, atomic across
  replicas, with a per-process fallback
- Image validation: size, format from magic bytes, decompression-bomb guard,
  SSRF protection on `image_url`
- Async batch processing through Celery
- Model versioning with per-request pinning and hot reload
- Graceful degradation with an explicit `degraded` flag
- Structured JSON logging with correlation IDs end to end
- One error envelope for every failure

### Part 3, testing

- 1,181 tests: 995 unit, 143 integration, 28 end-to-end, 15 performance, plus Locust load
  tests
- The unit suite runs with no external services, using fakeredis, in-memory
  SQLite and fake runtimes, so a fresh clone needs nothing installed
- Integration tests use the real thing: real ONNX artifacts, and real
  PostgreSQL and Redis when reachable. That is what catches a
  dialect-specific query or a Lua script that is not actually atomic
- End-to-end tests drive the deployed stack over HTTP, so they cross nginx,
  the container image, Redis, PostgreSQL and the Celery worker. That is what
  catches a container serving a stale artifact, which every in-process test
  passes straight over
- Every integration and end-to-end test skips cleanly when its dependency is
  missing, including unfetched Git LFS pointers, naming the remedy
- Memory profiling and concurrency verification
- CI with lint, type check, tests, a coverage gate, Docker build and a
  security scan

### Part 4, containerisation

- Multi-stage builds, non-root user (uid 10001), slim base images
- Seven services, all with health checks
- Production overlay: replicas, no exposed ports, read-only root filesystems,
  rolling updates, and secrets that are required rather than defaulted
- Three segmented networks, DNS service discovery, no hardcoded IPs
- Prometheus with 11 alert rules; Grafana auto-provisioned with a 22-panel
  dashboard

---

## Project layout

```text
├── api/                      FastAPI application
│   ├── main.py               app factory, lifespan, middleware stack
│   ├── config.py             settings; fails fast if insecure
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
│   ├── cards/                one card per model
│   ├── registry.py           model registry CLI
│   └── artifacts/            .onnx / .pt files, tracked in Git LFS
├── worker/                   Celery app and batch tasks
├── db/                       SQLAlchemy models
├── tests/                    unit, integration, performance
├── notebooks/                colab_gpu_pipeline.ipynb (generated)
├── Dockerfile                the API container
├── docker/                   Dockerfile.worker, nginx/
├── monitoring/               prometheus config + alerts, grafana provisioning
├── scripts/                  prepare_models, download_datasets, checklist
├── samples/                  three photos so the doc examples run as written
├── benchmarks/               baselines and generated reports
├── docs/                     API, TECHNICAL, ASSUMPTIONS, DEPLOYMENT, openapi
├── .gitattributes            Git LFS rules
└── .github/workflows/        CI pipeline
```

Two paths are generated rather than hand-edited:

- `notebooks/colab_gpu_pipeline.ipynb` comes from
  `scripts/build_colab_notebook.py`. Editing the notebook directly gets
  overwritten on the next build, and CI fails if the two drift apart.
- `DELIVERABLES_CHECKLIST.xlsx` comes from `scripts/generate_checklist.py`.

`models/artifacts/` is tracked with Git LFS. Those files cannot be regenerated
without a GPU session, unlike the dataset, which one command rebuilds. A clone
made without git-lfs gets 130-byte pointer files, and the model tests skip
with `git lfs pull` as the stated remedy.

Two files extend the brief's prescribed structure: `api/config.py`, required
by "environment-based configuration" and "no hardcoded secrets", and
`api/dependencies.py`, so image extraction is defined once rather than per
router. The extra routers exist because the brief requires those endpoints.

---

## Development

Activate the virtual environment from step 2 first.

```bash
pip install -r requirements-dev.txt

pytest tests/ -v                           # everything except performance
pytest tests/ --cov=api --cov-report=html  # with a coverage report
pytest tests/performance -m performance -s # timing and memory
ruff check api/ models/ worker/ tests/     # lint
black api/ models/ worker/ tests/          # format
```

Integration and end-to-end tests run automatically when `docker compose up -d`
is running, and skip when it is not:

```bash
pytest tests/e2e -v        # drives the deployed stack over HTTP
```

The status table at the top of this file is checked in CI against the actual
suite, so the numbers cannot quietly rot:

```bash
python scripts/check_readme_stats.py         # report drift
python scripts/check_readme_stats.py --fix   # update the README
```

### Working with models

```bash
python scripts/prepare_models.py                   # export and register
python -m models.registry list                     # what is registered
python -m models.registry validate                 # do the artifacts exist?
python -m models.validation.validate               # full validation suite
python -m models.optimization.benchmark            # latency across formats
python -m models.validation.regression check --model resnet50:1.0.0
```

### Training

Training always runs on the complete dataset: all 200 classes, all 100,000
training and 10,000 validation images, every batch. There is no option to
subset it. `verify_full_dataset` counts what is on disk, compares it against
what the dataloaders picked up, and refuses to train if anything is missing.

```bash
# The standard run. Uses CUDA when a GPU is available.
python -m models.training.train_classifier --epochs 30 --batch-size 256 --device auto

# Faster on CPU at some accuracy cost, still all 200 classes.
python -m models.training.train_classifier --arch resnet18 --no-stem-adapt --epochs 10
```

Measured CPU cost on an Intel Core Ultra 7 155H (16 threads):

| Config | img/s | 1 epoch | 30 epochs |
| --- | ---: | ---: | ---: |
| resnet50 + 64px stem | 3.8 | 7.6 h | 9.4 days |
| resnet50, original stem | 21.2 | 1.4 h | 1.7 days |
| resnet18 + 64px stem | 9.2 | 3.1 h | 3.9 days |
| resnet18, original stem | 75.0 | 23 min | 11.5 h |

The 64px stem adaptation dominates the cost: it is correct for 64px input, but
every layer after it then runs at 16× the spatial area. A GPU is roughly
30-60× faster.

For a long run on a hosted GPU, pass `--mirror-dir` pointing at mounted cloud
storage so checkpoints outlive the container.

---

## Security

- No hardcoded secrets. Production refuses to start without them, verified in
  CI.
- Authentication fails closed. No keys configured means every request is
  rejected; there is no default credential.
- Constant-time key comparison. Keys never appear in logs, only a
  non-reversible fingerprint.
- SSRF protection on `image_url`: scheme and port allow-lists, redirects
  disabled, and every resolved address checked against private, loopback and
  link-local ranges.
- Decompression-bomb guard: headers are parsed and pixel counts checked before
  any pixel buffer is allocated.
- Format detected from magic bytes, never from a filename or a client-declared
  content type.
- TorchScript rather than pickle, because loading a pickled checkpoint would
  execute arbitrary code from the artifact.
- JWT algorithm pinning, so an `alg: none` forgery is rejected.
- Non-root containers, read-only root filesystems in production,
  `no-new-privileges`.
- No images stored, only a SHA-256 hash.

---

## Known limitations

The full list with reasoning is in [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) §2.6.

1. **TensorRT INT8 is not built**, though fp32 and fp16 are. In TensorRT 11 an
   INT8 engine needs a QDQ graph; `quantize.py` produces one but it has not
   been run, so `precision="int8"` refuses rather than silently building fp32
   and labelling it INT8.
2. **Accuracy for the ImageNet-1k and COCO models is cited, not re-measured.**
   That needs the ImageNet and COCO validation sets. Behavioural correctness
   was verified end to end.
3. **The similarity index is per-process**, so it does not survive horizontal
   scaling. Options are in TECHNICAL.md.
4. **Confidence is not calibrated.** The fine-tuned classifier measures ECE
   0.1244 and the ImageNet-1k model 0.22. Use the ranking rather than the
   absolute scores unless you have measured otherwise on your own data.
5. **YOLOv8 is AGPL-3.0**, which matters for commercial use.
6. **The gateway round-robins rather than least-connections.** nginx caches an
   upstream's DNS answer at startup, so an `upstream` block pointed at a
   container that is later rebuilt keeps calling a dead IP. The fix resolves
   per request through a variable, which cannot reference an upstream block,
   so `least_conn` and upstream keepalive were given up to get self-healing.
   Reasoning is in `docker/nginx/nginx.conf`.

### Next, in priority order

1. Calibrate confidence with temperature scaling
2. Move the similarity index to a shared store (pgvector or FAISS)
3. Build a TensorRT INT8 engine from the existing QDQ graph
4. Measure accuracy against the real ImageNet and COCO validation sets
5. Add OpenTelemetry tracing
6. Restore least-connections balancing at the gateway
7. Alert when the rate limiter is running on local buckets rather than Redis
