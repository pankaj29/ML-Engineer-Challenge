# API Reference

Complete reference for the Multi-Model Computer Vision API.

Interactive documentation is generated from the code and served live:

| | |
| --- | --- |
| **Swagger UI** | http://localhost:8000/docs |
| **ReDoc** | http://localhost:8000/redoc |
| **OpenAPI spec** | http://localhost:8000/openapi.json, also committed at [`openapi.json`](openapi.json) |

The committed spec can be imported directly into Postman or Insomnia
(*Import → File → `docs/openapi.json`*), which generates a full request
collection.

---

## Contents

0. [Before you start: bash or PowerShell](#before-you-start-bash-or-powershell)
1. [Authentication](#1-authentication)
2. [Rate limits](#2-rate-limits)
3. [Supplying an image](#3-supplying-an-image)
4. [Endpoints](#4-endpoints)
5. [Errors](#5-errors)
6. [Correlation IDs](#6-correlation-ids)
7. [Model versioning](#7-model-versioning)

---

## Before you start: bash or PowerShell

Examples in this document are written for **bash** (macOS / Linux). They do
**not** work in Windows PowerShell, for three reasons:

| | |
| --- | --- |
| `curl` | In PowerShell this is an **alias for `Invoke-WebRequest`**, a different program that rejects `-X` and `-H`. Use `curl.exe` for real curl. |
| `\` at end of line | Bash line continuation. PowerShell uses a backtick `` ` ``. |
| `base64` | A Unix command. Windows has no equivalent binary. |

### PowerShell helpers

Paste these into your session once. Every PowerShell example below then fits
on one line.

```powershell
# Windows PowerShell
function Get-ImageB64 {
    param([string]$Path)
    [Convert]::ToBase64String([IO.File]::ReadAllBytes((Resolve-Path $Path).Path))
}

function Invoke-Api {
    param(
        [string]$Path,
        [hashtable]$Body,
        [string]$Method = "Post",
        [string]$Key = "dev-key-pro"
    )
    $req = @{
        Uri     = "http://localhost/api/v1/$Path"
        Method  = $Method
        Headers = @{ "X-API-Key" = $Key }
    }
    if ($Body) {
        $req.ContentType = "application/json"
        $req.Body = ($Body | ConvertTo-Json -Depth 6)
    }
    Invoke-RestMethod @req
}
```

Check they loaded:

```powershell
# Windows PowerShell
Invoke-Api health -Method Get      # -> status : healthy
```

`Get-ImageB64` needs `Resolve-Path`. `[IO.File]` is a .NET call and
resolves relative paths against .NET's own current directory, which `cd` does
**not** update - so a bare relative path fails with "Could not find file"
naming a folder you are not in.

### One call per task, in PowerShell

`samples/dog.jpg` ships with the repository, so these run as written. Any
image of your own works too.

```powershell
# Windows PowerShell
$img = Get-ImageB64 "samples\dog.jpg"

# classification
(Invoke-Api classify @{ image_base64 = $img; top_k = 5 }).predictions |
    Format-Table rank, label, confidence -AutoSize

# detection
(Invoke-Api detect @{ image_base64 = $img; confidence_threshold = 0.25 }).detections |
    Format-Table rank, label, confidence -AutoSize

# similarity - index one image, then search for it
Invoke-Api similarity/index  @{ image_base64 = $img; label = "example" }
(Invoke-Api similarity/search @{ image_base64 = $img; top_k = 3 }).results |
    Format-Table rank, label, score -AutoSize

# embedding vector only
(Invoke-Api similarity/embed @{ image_base64 = $img }).dimension     # 2048

# batch: submit, then poll
$job = Invoke-Api batch @{
    task  = "classification"
    top_k = 3
    items = @(@{ image_base64 = $img; image_id = "img-a" })
}
do {
    Start-Sleep -Seconds 2
    $s = Invoke-Api "batch/$($job.job_id)" -Method Get
    "$($s.status)  $($s.completed_items)/$($s.total_items)  failed=$($s.failed_items)"
} while ($s.status -notin @("completed", "failed", "cancelled"))

$s | ConvertTo-Json -Depth 5        # the full response
```

Multipart upload needs real curl, so use `curl.exe`:

```powershell
# Windows PowerShell - note curl.exe, not curl
curl.exe -X POST http://localhost/api/v1/classify/upload `
         -H "X-API-Key: dev-key-pro" `
         -F "file=@samples/dog.jpg" -F "top_k=5"
```

---

## 1. Authentication

Every endpoint requires credentials except `/health*`, `/metrics`, `/docs`,
`/redoc`, `/openapi.json`, `/auth/token` and `/`.

Two methods are accepted.

### API key (recommended for server-to-server)

```bash
curl -H "X-API-Key: your-key-here" http://localhost/api/v1/models
```

The key determines your **tier**, which determines your rate limit and maximum
batch size. Keys are configured server-side via the `API_KEYS` environment
variable as `key:tier` pairs.

### Bearer token (JWT)

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." http://localhost/api/v1/models
```

Short-lived, self-describing, and carries scopes. Tokens are verified with a
pinned algorithm, a token declaring `"alg": "none"` is rejected.

### Getting a token

`POST /api/v1/auth/token` trades an API key for one. The key is the long-lived
secret and belongs on a server; the token expires, so it is the safer thing to
hand to a browser or a mobile client.

```bash
curl -X POST http://localhost/api/v1/auth/token   -H "Content-Type: application/json"   -d '{"api_key": "dev-key-pro"}'
```

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600,
  "tier": "pro",
  "scopes": []
}
```

The token carries the key's own tier, so a free-tier key issues a free-tier
token with the free-tier rate limits. It cannot be used to escalate.

Pass `scopes` to restrict it further. Scopes only ever narrow, so a scoped
token can do less than the key, never more:

```bash
curl -X POST http://localhost/api/v1/auth/token   -H "Content-Type: application/json"   -d '{"api_key": "dev-key-pro", "scopes": ["read"]}'
```

That token is refused by `POST /models/reload`, which requires `admin`.

This endpoint is public, because it is how a caller obtains credentials in the
first place. It is not unauthenticated: the body carries an API key, which is
verified before anything is minted. A real deployment replaces it with an
identity provider and the contract stays the same.

### Failures

| Situation | Status | `code` |
| --- | --- | --- |
| No credentials | 401 | `AUTHENTICATION_FAILED` |
| Unknown API key | 401 | `AUTHENTICATION_FAILED` |
| Expired token | 401 | `AUTHENTICATION_FAILED` (`details.reason = "expired"`) |
| Valid, but insufficient scope | 403 | `PERMISSION_DENIED` |

> **Security note.** Keys are compared in constant time, and never appear in
> logs, only a short non-reversible fingerprint does.

---

## 2. Rate limits

Applied per API key, per minute, using a token bucket that allows a short
burst (1.5x the per-minute rate by default).

| Tier | Requests/minute | Max batch size |
| --- | ---: | ---: |
| `free` | 10 | 5 |
| `basic` | 60 | 16 |
| `pro` | 300 | 32 |
| `enterprise` | 3000 | 64 |

Every response carries your current allowance, so a well-behaved client
can slow down before being blocked:

```
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 287
X-RateLimit-Tier: pro
```

A rejection returns **429** with a `Retry-After` header:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "You have exceeded the 300 requests/minute allowance for the 'pro' tier. Retry in 3 seconds.",
    "details": { "limit_per_minute": 300, "tier": "pro", "retry_after_seconds": 3 }
  }
}
```

> **Batches cost more than one token.** A 20-image batch consumes 20 tokens.
> Without that, batching would be a trivial way around a tier limit.

There is a second, coarser limit at the Nginx gateway (30 req/s per IP on
inference paths). That one protects the infrastructure from a flood; the
per-tier limit above enforces fairness between customers.

---

## 3. Supplying an image

Every inference endpoint accepts an image three ways. Use whichever suits your
client.

### a. Base64 in a JSON body

```json
{ "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..." }
```

A `data:` URI prefix is accepted and stripped, so
`"data:image/png;base64,iVBOR..."` also works.

### b. A URL

```json
{ "image_url": "https://example.com/photo.jpg" }
```

The server fetches it. **Security:** the URL is validated before fetching , 
only `http`/`https`, only standard ports, redirects disabled, and any hostname
resolving to a private, loopback or link-local address is refused. This
prevents the endpoint being used to reach internal services or cloud metadata.

### c. Multipart upload

Every endpoint has an `/upload` variant:

```bash
curl -X POST http://localhost/api/v1/classify/upload \
     -H "X-API-Key: your-key" \
     -F "file=@cat.jpg" -F "top_k=3"
```

> Send **exactly one** source. Supplying both `image_base64` and `image_url`
> is a 422.

### Image limits

| Constraint | Default |
| --- | --- |
| Maximum size | 10 MB |
| Maximum pixels | 50,000,000 (decompression-bomb guard) |
| Dimensions | 16 px to 8192 px per side |
| Formats | JPEG, PNG, WEBP, BMP |

Format is determined from the file's **magic bytes**, never from its filename
or declared content type. Greyscale and RGBA images are accepted and converted
to RGB (alpha is composited onto white, not discarded). EXIF orientation is
applied, so a portrait phone photo is not analysed sideways.

---

## 4. Endpoints

Every endpoint at a glance:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/v1/classify` | POST | Classify one image (JSON body) |
| `/api/v1/classify/upload` | POST | Same, multipart file upload |
| `/api/v1/detect` | POST | Detect objects, returns boxes |
| `/api/v1/detect/upload` | POST | Same, multipart |
| `/api/v1/similarity/embed` | POST | Return the 2048-dim vector only |
| `/api/v1/similarity/index` | POST | Add an image to the search index |
| `/api/v1/similarity/search` | POST | Find nearest neighbours |
| `/api/v1/similarity/upload` | POST | Search by multipart upload |
| `/api/v1/similarity/stats` | GET | Index size, dimension, memory |
| `/api/v1/batch` | POST | Submit a background job, returns a job id |
| `/api/v1/batch/{job_id}` | GET | Job status, progress and results |
| `/api/v1/batch/{job_id}` | DELETE | Attempt to cancel a job |
| `/api/v1/models` | GET | List registered models; filter with `?task=` |
| `/api/v1/models/{name}` | GET | One model's metadata and metrics |
| `/api/v1/models/reload` | POST | Re-read `registry.json` without restarting |
| `/api/v1/health` | GET | Full check: models, cache, database |
| `/api/v1/health/live` | GET | Is the process alive? Checks no dependencies |
| `/api/v1/health/ready` | GET | Ready for traffic? Checks dependencies |
| `/api/v1/metrics` | GET | Prometheus metrics (private networks only) |

The two health probes answer different questions. `live` checks nothing external, so a
Redis hiccup cannot make the orchestrator restart healthy containers; `ready`
checks dependencies, so a degraded instance leaves the load balancer without
being killed.


### `POST /api/v1/classify`

Identify what an image contains.

**Request**

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `image_base64` / `image_url` | string |, | One is required |
| `top_k` | int 1-100 | 5 | How many classes to return |
| `confidence_threshold` | float 0-1 | 0.0 | Drop predictions below this |
| `include_probabilities` | bool | true | Include confidence values |
| `image_id` | string | null | Echoed back; useful for batches |
| `model_name` / `model_version` | string | null | Pin a specific model |
| `runtime` | enum | null | `onnx`, `onnx_int8`, `torch`, `tensorrt` |

```bash
curl -X POST http://localhost/api/v1/classify \
     -H "X-API-Key: dev-key-pro" -H "Content-Type: application/json" \
     -d @- <<EOF
{"image_base64": "$(base64 -w0 samples/dog.jpg)", "top_k": 3}
EOF
```

**Response `200`**

```json
{
  "predictions": [
    { "class_id": 208, "label": "Labrador retriever",    "confidence": 0.397, "rank": 1 },
    { "class_id": 205, "label": "flat-coated retriever", "confidence": 0.017, "rank": 2 },
    { "class_id": 227, "label": "kelpie",                "confidence": 0.014, "rank": 3 }
  ],
  "top_prediction": { "class_id": 208, "label": "Labrador retriever", "confidence": 0.397, "rank": 1 },
  "model":  { "name": "resnet50", "version": "1.0.0", "task": "classification",
              "runtime": "onnx", "device": "cpu" },
  "timing": { "preprocess_ms": 12.4, "inference_ms": 69.9,
              "postprocess_ms": 0.3, "total_ms": 82.6 },
  "correlation_id": "3803dbb1d6274a2e9f1c...",
  "cached": false,
  "image_id": null,
  "degraded": false,
  "warnings": ["image resized from 640x480 to 224x224 using center_crop"]
}
```

`degraded: true` means the requested model was unavailable and a fallback
served the request. `cached: true` means the result came from Redis.

> **On confidence:** these scores are *not* calibrated probabilities. A 0.9 does
> not mean "right 90% of the time". Use the ranking, which is reliable; see the
> model card before thresholding on the absolute value.

---

### `POST /api/v1/detect`

Find objects and where they are.

**Request**

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `confidence_threshold` | float 0-1 | 0.25 | Minimum score for a box |
| `iou_threshold` | float 0-1 | 0.45 | Overlap above which a box is a duplicate |
| `max_detections` | int 1-300 | 100 | Cap on boxes returned |
| `class_filter` | string[] | null | Only return these class names |

**Response `200`**

```json
{
  "detections": [
    { "class_id": 0, "label": "person", "confidence": 0.647,
      "box": { "x1": 157.9, "y1": 40.1, "x2": 522.0, "y2": 423.1 } }
  ],
  "count": 1,
  "image_width": 640,
  "image_height": 480,
  "model": { "name": "yolov8n", "version": "1.0.0", "task": "detection",
             "runtime": "onnx", "device": "cpu" },
  "timing": { "preprocess_ms": 18.2, "inference_ms": 97.1,
              "postprocess_ms": 4.1, "total_ms": 142.9 },
  "correlation_id": "...", "cached": false
}
```

Coordinates are absolute pixels in the image you uploaded, with `x1,y1`
top-left and `x2,y2` bottom-right. You can draw them directly without knowing
anything about the model's internal resize.

Tuning the thresholds:

* Raise `confidence_threshold` to cut false positives; lower it to catch more
  objects (at the cost of noise).
* Raise `iou_threshold` to keep genuinely overlapping objects in a crowd;
  lower it to suppress duplicate boxes. No single value is right for every
  scene.

---

### `POST /api/v1/similarity/embed`

Turn an image into a vector, without storing anything. Use this if you keep
your own vector store.

```json
{ "embedding": [0.0134, -0.0721, ...], "dimension": 2048, "model": {...}, "timing": {...} }
```

The vector is L2-normalised, so the **dot product of two embeddings is their
cosine similarity**.

### `POST /api/v1/similarity/index` → `201`

Embed an image and store it so future searches can find it.

| Field | Type | Description |
| --- | --- | --- |
| `label` | string | Human-readable name, returned with any hit |
| `metadata` | object | Arbitrary JSON, returned with any hit |
| `image_id` | string | Supply your own id; one is generated otherwise |

```json
{ "id": "a3f2...", "index_size": 1, "correlation_id": "..." }
```

### `POST /api/v1/similarity/search`

Find the indexed images most similar to the supplied one.

| Field | Type | Default |
| --- | --- | --- |
| `top_k` | int 1-100 | 10 |
| `min_similarity` | float -1 to 1 | 0.0 |
| `include_embedding` | bool | false |

```json
{
  "results": [
    { "id": "a3f2...", "score": 1.0, "rank": 1, "label": "the bus photo", "metadata": null }
  ],
  "count": 1,
  "index_size": 1,
  "timing": { "preprocess_ms": 9.1, "inference_ms": 49.6, "postprocess_ms": 0.1, "total_ms": 53.2 }
}
```

`postprocess_ms` here is the **vector search time**, kept separate so you can
see how much latency is the model and how much is the search.

Scores run from `1.0` (identical direction) through `0.0` (unrelated) to
`-1.0` (opposite). Near-duplicates typically exceed 0.95; "same kind of thing"
is roughly 0.7-0.9. **Tune the threshold on your own images**, there is no
universal value.

An empty `results` with `index_size: 0` means nothing has been indexed yet,
not that nothing matched.

### `GET /api/v1/similarity/stats`

Index size, dimensionality and memory use.

---

### `POST /api/v1/batch` → `202 Accepted`

Submit many images for background processing.

```json
{
  "task": "classification",
  "items": [
    { "image_base64": "...", "image_id": "photo-1" },
    { "image_url": "https://example.com/b.jpg", "image_id": "photo-2" }
  ],
  "top_k": 5,
  "callback_url": "https://your-app.example.com/webhook",
  "priority": 5
}
```

`task` is `classification`, `detection` or `similarity`. `callback_url` must
be HTTPS.

**Response `202`**

```json
{
  "job_id": "d084d1fc-...",
  "status": "pending",
  "task": "classification",
  "total_items": 4,
  "status_url": "/api/v1/batch/d084d1fc-...",
  "estimated_seconds": 0.5,
  "correlation_id": "...",
  "submitted_at": "2026-09-22T15:20:11Z"
}
```

202 means **accepted**, not finished. Poll `status_url`. Only the API key
that submitted a job can read or cancel it; any other key gets 404, so a job id
alone reveals nothing.

### `GET /api/v1/batch/{job_id}`

```json
{
  "job_id": "d084d1fc-...",
  "status": "completed",
  "total_items": 4,
  "completed_items": 3,
  "failed_items": 1,
  "progress_percent": 100.0,
  "duration_seconds": 11.21,
  "results": [
    { "index": 0, "image_id": "good-1", "success": true,
      "result": { "predictions": [...] }, "duration_ms": 6573.0 },
    { "index": 1, "image_id": "BAD", "success": false, "result": null,
      "error": { "code": "INVALID_IMAGE", "message": "The image does not look like an image file." },
      "duration_ms": 2.1 }
  ]
}
```

> **One bad image does not fail the batch.** Each item carries its own
> `success` and `error`. A job can be `completed` with `failed_items > 0` , 
> that is normal, and far more useful than a single top-level error that tells
> you nothing about which image was the problem.

`status` moves `pending` → `running` → `completed` | `failed`, with
`progress_percent` updating after every image.

Add `?include_results=false` to poll progress without transferring results.

### `DELETE /api/v1/batch/{job_id}`

Cancel a queued or running job. A running job is terminated, so completed
items are **not** retrievable afterwards. A finished job cannot be cancelled
and returns `cancelled: false` with a reason.

---

### `GET /api/v1/models`

Discover what is available.

Query: `?task=classification` to filter, `?loaded_only=true` for
currently-resident models.

```json
{
  "models": [
    {
      "name": "resnet50", "version": "1.0.0", "task": "classification",
      "runtime": "onnx", "device": "cpu", "loaded": true, "is_default": true,
      "num_classes": 1000, "input_shape": [1, 3, 224, 224],
      "metrics": { "p50_latency_ms": 69.9, "size_mb": 97.4 },
      "description": "resnet50 pretrained on ImageNet-1k, exported to ONNX...",
      "limitations": [
        "Trained on ImageNet-1k: only recognises those 1000 categories...",
        "Confidence is not calibrated, a 0.9 score does not mean 90% correct."
      ]
    }
  ],
  "count": 3,
  "defaults": {
    "classification": "resnet50:1.0.0",
    "detection": "yolov8n:1.0.0",
    "similarity": "resnet50-embed:1.0.0"
  }
}
```

The `limitations` come straight from the model cards, so the caveats travel
with the model, so it cannot rot in a separate document.

### `GET /api/v1/models/{name}`, one model, `?version=` to pin.

### `POST /api/v1/models/reload`, re-read the registry without a restart.

Requires the `admin` scope. Also invalidates cached results for any model that
was removed, so predictions from retired weights cannot keep being served.

---

### Health

| Endpoint | Purpose | Use it for |
| --- | --- | --- |
| `GET /health/live` | Is the process alive? | Container restart policy |
| `GET /health/ready` | Can it serve right now? | Load-balancer membership |
| `GET /health` | Full dependency status | Dashboards, humans |

These are different, and conflating them causes outages.
Liveness checks nothing external. If it depended on the
database, a brief database blip would restart every container simultaneously
and turn a small problem into a total one.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "local",
  "uptime_seconds": 182.4,
  "components": [
    { "name": "cache",    "status": "healthy", "latency_ms": 1.2 },
    { "name": "database", "status": "healthy", "latency_ms": 3.4 },
    { "name": "model:classification", "status": "healthy" }
  ]
}
```

`healthy` and `degraded` both return **200**; only `unhealthy` returns 503.
A degraded instance can still serve traffic and should stay in the pool.

### `GET /api/v1/metrics`

Prometheus exposition format. Unauthenticated so Prometheus can scrape it, but
the gateway restricts it to internal networks.

---

## 5. Errors

Every error, from any endpoint, uses one envelope:

```json
{
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "The uploaded image is 14.2 MB, which exceeds the 10 MB limit.",
    "details": { "size_bytes": 14889000, "limit_bytes": 10485760, "field": "image" },
    "correlation_id": "0f6c1d8a...",
    "timestamp": "2026-09-22T15:07:58Z"
  }
}
```

Branch on `code`, never on `message`. Codes are stable, messages are
written for humans and may be reworded.

| Status | Code | Meaning | What to do |
| --- | --- | --- | --- |
| 400 | `INVALID_IMAGE` | Not a decodable image | Check the file and encoding |
| 401 | `AUTHENTICATION_FAILED` | Missing/invalid credentials | Check your key |
| 403 | `PERMISSION_DENIED` | Insufficient scope or tier | Upgrade or request access |
| 404 | `MODEL_NOT_FOUND` | No such model/version | `GET /models` for what exists |
| 404 | `JOB_NOT_FOUND` | Unknown job id, or a job submitted by another key | Per-item results expire after 24 h; the job's status and counts do not |
| 413 | `IMAGE_TOO_LARGE` | Over size/pixel limit | Resize before sending |
| 413 | `BATCH_TOO_LARGE` | Over your tier's batch cap | Split it, or upgrade |
| 415 | `UNSUPPORTED_FORMAT` | Format not allowed | Convert to JPEG or PNG |
| 422 | `VALIDATION_ERROR` | Body failed validation | See `details.fields` |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Wait `Retry-After` seconds |
| 500 | `INFERENCE_FAILED` | Model failed | Retry; quote the correlation id |
| 503 | `MODEL_LOAD_FAILED` | Model not ready | Retry shortly |
| 503 | `SERVICE_UNAVAILABLE` | A dependency is down | Retry with backoff |
| 503 | `SERVICE_OVERLOADED` | At capacity | Back off and retry |
| 504 | `INFERENCE_TIMEOUT` | Exceeded its deadline | Try a smaller image |

Internal details are never returned. Stack traces, file paths and SQL go
to the logs, keyed by the correlation id. Quote that id to support.

### Retry guidance

| Status | Retry? |
| --- | --- |
| 4xx (except 429) | **No**, fix the request first |
| 429 | Yes, after `Retry-After` |
| 500, 503, 504 | Yes, with exponential backoff and jitter |

---

## 6. Correlation IDs

Every response carries `X-Correlation-ID`. That id appears on every log line
produced while handling the request, in the worker if it became a batch job,
and on the inference record in PostgreSQL.

Supply your own to trace a request across your system and ours:

```bash
curl -H "X-Correlation-ID: my-trace-abc-123" ...
```

If you supply one it is kept end to end. It must be 1 to 64 characters of
letters, digits, `.`, `_`, `:` or `-`; anything else is replaced with a fresh
id, so a malformed header cannot break the log record or inject text into
the logs.

---

## 7. Model versioning

By default you get each task's current default model. To pin:

```json
{ "image_base64": "...", "model_name": "resnet50", "model_version": "1.0.0" }
```

`model_version: "latest"` (or omitting it) takes the highest version of that
model.

A typo gives a 404, never a silent substitution. Asking for a model that does
not exist returns `MODEL_NOT_FOUND`. Being handed predictions from a different
model than the one you asked for is worse than an error.

The fallback that *does* exist is for **failures**, not typos: if a registered
model cannot be loaded, the task default serves the request and the response
is marked `degraded: true`. Degraded results are never cached, so they cannot
outlive the incident.

### Selecting a runtime

```json
{ "image_base64": "...", "runtime": "onnx_int8" }
```

`onnx` (float32) is the default. INT8 is ~4x smaller but **measured slower**
on CPUs without INT8 acceleration, see
[`../benchmarks/reports/BENCHMARKS.md`](../benchmarks/reports/BENCHMARKS.md)
before switching.
