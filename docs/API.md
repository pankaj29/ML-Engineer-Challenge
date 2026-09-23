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

1. [Authentication](#1-authentication)
2. [Rate limits](#2-rate-limits)
3. [Supplying an image](#3-supplying-an-image)
4. [Endpoints](#4-endpoints)
5. [Errors](#5-errors)
6. [Correlation IDs](#6-correlation-ids)
7. [Model versioning](#7-model-versioning)

---

## 1. Authentication

Every endpoint requires credentials except `/health*`, `/metrics`, `/docs`,
`/redoc`, `/openapi.json` and `/`.

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
pinned algorithm — a token declaring `"alg": "none"` is rejected.

### Failures

| Situation | Status | `code` |
| --- | --- | --- |
| No credentials | 401 | `AUTHENTICATION_FAILED` |
| Unknown API key | 401 | `AUTHENTICATION_FAILED` |
| Expired token | 401 | `AUTHENTICATION_FAILED` (`details.reason = "expired"`) |
| Valid, but insufficient scope | 403 | `PERMISSION_DENIED` |

> **Security note.** Keys are compared in constant time, and never appear in
> logs — only a short non-reversible fingerprint does.

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

**Every response** carries your current allowance, so a well-behaved client
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

The server fetches it. **Security:** the URL is validated before fetching —
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

### `POST /api/v1/classify`

Identify what an image contains.

**Request**

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `image_base64` / `image_url` | string | — | One is required |
| `top_k` | int 1-100 | 5 | How many classes to return |
| `confidence_threshold` | float 0-1 | 0.0 | Drop predictions below this |
| `include_probabilities` | bool | true | Include confidence values |
| `image_id` | string | null | Echoed back; useful for batches |
| `model_name` / `model_version` | string | null | Pin a specific model |
| `runtime` | enum | null | `onnx`, `onnx_int8`, `torch`, `tensorrt` |

```bash
curl -X POST http://localhost/api/v1/classify \
     -H "X-API-Key: dev-key-pro" -H "Content-Type: application/json" \
     -d '{"image_base64": "'"$(base64 -w0 photo.jpg)"'", "top_k": 3}'
```

**Response `200`**

```json
{
  "predictions": [
    { "class_id": 654, "label": "minibus",    "confidence": 0.190, "rank": 1 },
    { "class_id": 874, "label": "trolleybus", "confidence": 0.081, "rank": 2 },
    { "class_id": 829, "label": "streetcar",  "confidence": 0.061, "rank": 3 }
  ],
  "top_prediction": { "class_id": 654, "label": "minibus", "confidence": 0.190, "rank": 1 },
  "model":  { "name": "resnet50", "version": "1.0.0", "task": "classification",
              "runtime": "onnx", "device": "cpu" },
  "timing": { "preprocess_ms": 12.4, "inference_ms": 84.8,
              "postprocess_ms": 0.3, "total_ms": 97.5 },
  "correlation_id": "3803dbb1d6274a2e9f1c...",
  "cached": false,
  "image_id": null,
  "degraded": false,
  "warnings": ["image resized from 810x1080 to 224x224 using center_crop"]
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
    { "class_id": 0, "label": "person", "confidence": 0.90,
      "box": { "x1": 671.0, "y1": 385.0, "x2": 810.0, "y2": 880.0 } },
    { "class_id": 5, "label": "bus", "confidence": 0.84,
      "box": { "x1": 31.0, "y1": 231.0, "x2": 801.0, "y2": 778.0 } }
  ],
  "count": 2,
  "image_width": 810,
  "image_height": 1080,
  "model": { "name": "yolov8n", "version": "1.0.0", "task": "detection",
             "runtime": "onnx", "device": "cpu" },
  "timing": { "preprocess_ms": 18.2, "inference_ms": 120.6,
              "postprocess_ms": 4.1, "total_ms": 142.9 },
  "correlation_id": "...", "cached": false
}
```

**Coordinates are absolute pixels in the image you uploaded**, with `x1,y1`
top-left and `x2,y2` bottom-right. You can draw them directly without knowing
anything about the model's internal resize.

**Tuning the thresholds**

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
| `image_id` | string | Use your own id instead of a generated one |

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
  "timing": { "preprocess_ms": 9.1, "inference_ms": 43.4, "postprocess_ms": 0.1, "total_ms": 53.2 }
}
```

`postprocess_ms` here is the **vector search time**, kept separate so you can
see how much latency is the model and how much is the search.

Scores run from `1.0` (identical direction) through `0.0` (unrelated) to
`-1.0` (opposite). Near-duplicates typically exceed 0.95; "same kind of thing"
is roughly 0.7-0.9. **Tune the threshold on your own images** — there is no
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

202 means **accepted**, not finished. Poll `status_url`.

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
> `success` and `error`. A job can be `completed` with `failed_items > 0` —
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
      "metrics": { "p50_latency_ms": 84.8, "size_mb": 97.4 },
      "description": "resnet50 pretrained on ImageNet-1k, exported to ONNX...",
      "limitations": [
        "Trained on ImageNet-1k: only recognises those 1000 categories...",
        "Confidence is not calibrated — a 0.9 score does not mean 90% correct."
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
with the model rather than living in a document nobody reads.

### `GET /api/v1/models/{name}` — one model, `?version=` to pin.

### `POST /api/v1/models/reload` — re-read the registry without a restart.

Requires the `admin` scope. Also invalidates cached results for any model that
was removed, so predictions from retired weights cannot keep being served.

---

### Health

| Endpoint | Purpose | Use it for |
| --- | --- | --- |
| `GET /health/live` | Is the process alive? | Container restart policy |
| `GET /health/ready` | Can it serve right now? | Load-balancer membership |
| `GET /health` | Full dependency status | Dashboards, humans |

These are genuinely different, and conflating them causes outages.
**Liveness deliberately checks nothing external** — if it depended on the
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

**Every** error, from any endpoint, uses one envelope:

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

**Branch on `code`, never on `message`.** Codes are stable; messages are
written for humans and may be reworded.

| Status | Code | Meaning | What to do |
| --- | --- | --- | --- |
| 400 | `INVALID_IMAGE` | Not a decodable image | Check the file and encoding |
| 401 | `AUTHENTICATION_FAILED` | Missing/invalid credentials | Check your key |
| 403 | `PERMISSION_DENIED` | Insufficient scope or tier | Upgrade or request access |
| 404 | `MODEL_NOT_FOUND` | No such model/version | `GET /models` for what exists |
| 404 | `JOB_NOT_FOUND` | Unknown job id | Results expire after 24 h |
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

**Internal details are never returned.** Stack traces, file paths and SQL go
to the logs, keyed by the correlation id. Quote that id to support.

### Retry guidance

| Status | Retry? |
| --- | --- |
| 4xx (except 429) | **No** — fix the request first |
| 429 | Yes, after `Retry-After` |
| 500, 503, 504 | Yes, with exponential backoff and jitter |

---

## 6. Correlation IDs

Every response carries `X-Correlation-ID`. That id appears on every log line
produced while handling the request, in the worker if it became a batch job,
and on the inference record in PostgreSQL.

**Supply your own** to trace a request across your system and ours:

```bash
curl -H "X-Correlation-ID: my-trace-abc-123" ...
```

If you supply one it is preserved end to end, not replaced.

---

## 7. Model versioning

By default you get each task's current default model. To pin:

```json
{ "image_base64": "...", "model_name": "resnet50", "model_version": "1.0.0" }
```

`model_version: "latest"` (or omitting it) takes the highest version of that
model.

**A typo is a 404, not a silent substitution.** Asking for a model that does
not exist returns `MODEL_NOT_FOUND` rather than quietly serving something
else — being handed predictions from a different model than you asked for is
worse than an error.

The fallback that *does* exist is for **failures**, not typos: if a registered
model cannot be loaded, the task default serves the request and the response
is marked `degraded: true`. Degraded results are never cached, so they cannot
outlive the incident.

### Selecting a runtime

```json
{ "image_base64": "...", "runtime": "onnx_int8" }
```

`onnx` (float32) is the default. INT8 is ~4x smaller but **measured slower**
on CPUs without INT8 acceleration — see
[`../benchmarks/reports/BENCHMARKS.md`](../benchmarks/reports/BENCHMARKS.md)
before switching.
