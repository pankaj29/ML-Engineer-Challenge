# Assumptions, Decisions and Known Gaps

The brief says: *"Feel free to make reasonable assumptions and document them."*
This is that document.

It is organised by how much it should matter to a reviewer:

1. **[Ambiguities in the brief](#1-ambiguities-in-the-brief)** — where the
   requirements were open to reading, and which reading was taken.
2. **[Things not fully delivered](#2-things-not-fully-delivered)** — stated
   plainly, with the reason and the evidence.
3. **[Engineering decisions](#3-engineering-decisions)** — choices made where
   the brief was silent.
4. **[Bugs found in the provided scaffolding](#4-bugs-found-in-the-provided-scaffolding)**
5. **[Bugs found in my own work](#5-bugs-found-in-my-own-work)** — because
   how defects were caught says more than a clean-looking result.

---

## 1. Ambiguities in the brief

### 1.1 "Utilise 3 different models" — but only two are listed

The overview says the system serves *"image classification, object detection,
and image similarity search"*. The numbered requirement then says **"Utilise 3
different models"** and lists only two bullets. The prescribed API structure
lists only `classification.py` and `detection.py`.

**Assumption: the third model is image similarity search**, per the overview.
Confirmed with the requester before building.

**Delivered:** `resnet50` (classification), `yolov8n` (detection),
`resnet50-embed` (similarity), plus `/api/v1/similarity/*` endpoints.

### 1.2 The classification dataset: CIFAR-100 or Tiny-ImageNet?

The brief names both, in consecutive bullets:

> - Image Classification (ViT / ResNet / EfficientNet etc. **on CIFAR-100**)
> - **For Image classification only**: fine-tune the model with ... mixed
>   precision, gradient clipping, learning rate scheduling. Use the
>   **tiny-ImageNet dataset** for this.

These are different datasets. CIFAR-100 is 32x32 with 100 classes;
Tiny-ImageNet is 64x64 with 200. They need different input handling and
produce different models.

**Assumption: Tiny-ImageNet is the operative instruction.** Three reasons:

1. It is the *explicit* one ("Use the tiny-ImageNet dataset for this"), and it
   is attached to the fine-tuning requirement that carries the three
   techniques actually being assessed. The CIFAR-100 mention is a
   parenthetical sitting alongside a list of candidate architectures, which
   reads as an example rather than a mandate.
2. It is the only dataset the brief gives a download command for, and the one
   its Getting Started section fetches.
3. It is the harder and more representative task: 200 classes at 64x64
   exercises the pipeline more than 100 classes at 32x32.

**CIFAR-100 has not been abandoned.** `scripts/download_datasets.py` already
supports it (`--dataset cifar100`), and the training pipeline is
dataset-shaped rather than Tiny-ImageNet-specific: adding it means writing one
`Dataset` class alongside `models/training/dataset.py` and passing different
normalisation constants. If CIFAR-100 was the intent, that is a contained
change rather than a rewrite.

### 1.3 Two different paths given for the dataset script

The brief gives `./scripts/download_datasets.py` in Part 1 and
`scripts/setup/download_datasets.py` in Getting Started. The file exists at
the former.

**Assumption: the same script, documented inconsistently.** Left at its actual
location; the README uses the path that works.

### 1.4 Fine-tuning target vs the model that is served

Part 1 requires fine-tuning a classifier on **Tiny-ImageNet** with mixed
precision, gradient clipping and LR scheduling. Part 2 requires a production
API serving image classification.

These pull in different directions: a Tiny-ImageNet model knows 200 classes of
64x64 thumbnails and is close to useless as a general-purpose classification
API.

**Assumption: both are wanted, for different purposes.**

* The **fine-tuning pipeline** is complete, and demonstrates all three
  required techniques (`models/training/train_classifier.py`).
* The **served model** is ImageNet-1k ResNet-50, which is what a real
  classification API would serve.

Registering the fine-tuned checkpoint instead is a one-line registry change.

### 1.5 "COCO subset" for detection

**Assumption: pretrained COCO weights satisfy this.** They are trained on
COCO, which is a superset of any subset. Training a detector from scratch on a
COCO subset would produce a strictly worse model for more effort. The
`coco_sample` download (val2017) is available in the provided script if a
subset is genuinely wanted for evaluation.

### 1.6 Grafana marked "optional but preferred"

**Delivered**, with automatic provisioning of the data source and a 22-panel
dashboard. Verified live.

---

## 2. Things not fully delivered

### 2.1 TensorRT export is written but has never been run

**Requirement:** *"Convert models to ONNX and TensorRT formats."*

**Status: code complete, execution blocked.** TensorRT requires an NVIDIA GPU
and the `tensorrt` package; the development machine has neither (CPU-only
Intel Core Ultra 7 155H).

What exists:

* `models/optimization/export_tensorrt.py` — engine building for fp32/fp16/INT8,
  INT8 calibration, optimisation profiles, numerical verification against ONNX,
  and benchmarking.
* `TensorRTBackend` in `api/services/model_service.py` — a runtime that loads
  and executes an engine, wired into the same fallback chain as every other
  format.
* `requirements-gpu.txt`.

The code is gated: on a CPU host it prints a clear explanation and exits 0
(verified). **Treat it as untested until it has run on real hardware.** It is
written against the TensorRT 10.x API.

Run it on a GPU host with:

```bash
pip install -r requirements-gpu.txt
python -m models.optimization.export_tensorrt --onnx models/artifacts/resnet50.onnx --precision fp16 --benchmark
```

### 2.2 The fine-tuned classifier: trained in full, checkpoint not yet in the repo

**Requirement:** fine-tune on Tiny-ImageNet with mixed precision, gradient
clipping and LR scheduling.

**Status: pipeline complete and a full 30-epoch run completed.**

The run was done on an NVIDIA A100-SXM4-40GB (Google Colab, driven from VS Code
over the Colab connector), on all 200 classes and all 100,000 training images —
no subsetting.

| | |
| --- | --- |
| Epochs | 30 |
| Batch size | 256 |
| Hardware | A100-SXM4-40GB |
| Time per epoch | ~82 s |
| Final top-1 | **73.98%** |
| Final top-5 | **90.18%** |
| Random baseline | 0.5% top-1 |

All three required techniques were active during the run: `torch.autocast`
fp16 with `GradScaler`, `clip_grad_norm_` after unscaling, and cosine LR with
linear warmup.

**The gap:** the resulting checkpoint has not yet been pulled back into this
repo, so `models/artifacts/` still contains only the ImageNet-1k exports. The
accuracy figures above come from the training log, not from a re-run of
`models/validation/validate.py` against a committed checkpoint. Treat them as
reported-by-the-run until the checkpoint lands.

**Why CPU training was not an option.** Measured on the build machine
(Intel Core Ultra 7 155H, 16 threads), timing real training steps rather than
estimating:

| Config | img/s | 1 epoch | 30 epochs |
| --- | ---: | ---: | ---: |
| resnet50 + 64px stem (the default) | 3.8 | 7.6 h | **9.4 days** |
| resnet50, original stem | 21.2 | 1.4 h | 1.7 days |
| resnet18 + 64px stem | 9.2 | 3.1 h | 3.9 days |
| resnet18, original stem | 75.0 | 23 min | 11.5 h |

The 64px stem adaptation dominates. Replacing ResNet's stride-2 stem with a
stride-1 3x3 convolution is correct for 64px input — the original reduces a
64x64 image to 16x16 before the first residual block — but every layer after
it then runs at 16x the spatial area. On the A100 the same config finishes in
41 minutes.

```bash
python -m models.training.train_classifier --epochs 30 --batch-size 256 --device auto
```

**Subsetting has been removed.** The `--classes` and `--limit-batches` flags
used during development are gone, and `verify_full_dataset()` now refuses to
train unless every class and every image found on disk was loaded.

### 2.3 Accuracy figures are cited, not re-measured

Top-1/top-5 for ResNet-50 and mAP for YOLOv8n are the published figures for
those checkpoints. Re-measuring needs the ImageNet (~6 GB, account required)
and COCO (~1 GB) validation sets.

**What was verified instead:** that the models produce correct predictions on
real images end to end, that ONNX export is numerically faithful to PyTorch
(max diff < 4e-06), and that they pass every behavioural validation check.

The validation pipeline **refuses** to report accuracy when the model's label
space does not match the evaluation set's — see §5.2.

### 2.4 Similarity index is per-process, not shared

The vector index lives in one process's memory. With several API replicas,
each has its own index. Correct for a single instance; wrong for a scaled
deployment. Options are set out in `docs/TECHNICAL.md`.

### 2.5 Alembic is configured but no migrations are committed

Tables are created with `Base.metadata.create_all` outside production.
Production should use versioned migrations; the dependency is present and the
models are migration-ready, but no initial revision is committed.

---

## 3. Engineering decisions

### 3.1 INT8 is not the default runtime, despite being 4x smaller

The brief asks for INT8 quantization on all models. It is applied to all
three. It is **not** the serving default, because measurement showed it is
slower on this hardware:

| Variant | p50 (ResNet-50, batch 1) | Size |
| --- | ---: | ---: |
| ONNX fp32 | 75.7 ms | 97.4 MB |
| INT8 **dynamic** | 1008.0 ms | 24.5 MB |
| INT8 **static QDQ** | 104.6 ms | 24.9 MB |

Dynamic quantization was **13x slower than fp32**. It recomputes activation
scales on every call and falls back to poorly-optimised integer convolution
kernels — a bad fit for convolutional networks. Switching to static QDQ with
real calibration images made it ~10x faster than dynamic, though still ~1.4x
slower than fp32.

**Decision:** ship fp32 as the default, register INT8 alongside it, and let
callers select it per request. Shipping a 13x-slower "optimisation" as the
default because the brief said "apply quantization" would have been the wrong
call. Full reasoning in `docs/TECHNICAL.md`.

### 3.2 Two files added to the prescribed API structure

The brief's tree is followed exactly, plus:

* `api/config.py` — required by "no hardcoded secrets" and
  "environment-based configuration".
* `api/dependencies.py` — shared FastAPI dependencies, so that image
  extraction and validation are defined once rather than in each router.

Additional routers (`batch.py`, `models.py`, `health.py`, `metrics.py`,
`similarity.py`) exist because the brief requires those endpoints.

### 3.3 The rate limiter fails open; the cache fails soft; auth fails closed

Three deliberate and different choices:

* **Rate limiter fails OPEN.** If Redis is down, traffic is allowed (with a
  local per-process bucket as a partial backstop). A cache outage should not
  become a total outage.
* **Cache fails SOFT** — every failure degrades to a cache miss, so the system
  becomes slower, never wrong.
* **Authentication fails CLOSED.** If no API keys are configured, every
  request is rejected. There is no default credential.

### 3.4 Only the worker is async for batches; single images are synchronous

Single-image endpoints respond directly; batches return HTTP 202 and a job id.
The cut-off is that a batch can exceed any sensible HTTP timeout. A batch also
costs N rate-limit tokens rather than 1, so batching cannot be used to bypass
a tier limit.

### 3.5 Images are never stored

Only a SHA-256 hash of the bytes, plus dimensions and format. Enough for
caching, deduplication and drift monitoring; nothing retained that is
sensitive.

### 3.6 Tests run with no external services

Redis is replaced by `fakeredis`, PostgreSQL by in-memory SQLite, and models
by a `FakeRuntime`. A suite that needs infrastructure running is a suite that
gets skipped. Real-artifact tests exist separately and skip cleanly when the
artifacts are absent.

### 3.7 Data lives inside the repository

Requested explicitly. Noted at the time that the repository sits inside a
OneDrive-synced folder, so 120k dataset files will be synced; `data/` is
gitignored. Pausing OneDrive sync during dataset work is advisable.

---

## 4. Bugs found in the provided scaffolding

Three real defects in the starter code, all fixed or worked around:

### 4.1 `scripts/download_datasets.py` crashes on the documented command

```python
choices=list(DatasetDownloader({}).datasets.keys()) + ["all"]
```

`{}` is passed where a path is expected, so `Path({})` raises `TypeError`
before any argument is parsed. **The documented command could not run.**
Fixed by constructing the throwaway instance with the real default directory.

### 4.2 `scripts/tiny_imagenet_dataloader.py` mislabels the validation set

```python
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), ...)
```

Tiny-ImageNet's validation split is **not** in `ImageFolder` layout. It is a
flat `val/images/` directory plus a `val_annotations.txt` mapping file.
`ImageFolder` treats `images` as a single class and assigns **label 0 to all
10,000 validation images**.

This is the more dangerous of the two, because it does not crash. Validation
accuracy reads a meaningless ~0.5% and looks like a broken training loop.

Verified against the real dataset before fixing:
`ImageFolder on val/ -> 10000 images, 1 distinct label: {0: 10000}`.

**Fixed.** The script now reads `val_annotations.txt` and reuses the training
split's class-to-index mapping, so the two splits line up. Its public API is
unchanged, so any existing caller keeps working — it just gets correct labels.
A self-check in `__main__` fails loudly if the validation labels are ever
degenerate again.

Verified after fixing: **200 distinct labels, exactly 50 images each.**

(`models/training/dataset.py` has a more fully-featured loader — class
subsetting, readable class names — and is what the training pipeline uses.)

### 4.3 `scripts/sample_serving_script.py` is a synchronous Flask stub

Single-threaded, no validation, no error handling, model in the request path.
Treated as illustrative only; the delivered API shares none of it.

---

## 5. Bugs found in my own work

The checks that caught them are part of the deliverable, so what they caught
is worth stating.

### 5.1 ONNX export silently broke batching

torch 2.9's default "dynamo" exporter **ignored `dynamic_axes`**, baking in a
batch size of 1, and split weights into a sidecar `.onnx.data` file. Caught by
the export script's own dynamic-batch check. Fixed by opting out of the dynamo
exporter.

### 5.2 Validation reported a confident, meaningless 0% accuracy

Evaluating ResNet-50 (1,000 ImageNet classes) against Tiny-ImageNet labels
(200 classes) compared unrelated integers and produced 0.00% top-1 — which
looks exactly like a broken model. Fixed: the pipeline now detects a label
space mismatch and refuses to report accuracy, explaining why.

### 5.3 YOLO ONNX had a fixed batch dimension

Caught by the batch-invariance check. Would have made every batch detection
job fail. Fixed with `dynamic=True`.

### 5.4 A `# noqa` comment corrupted the Redis Lua script

A lint suppression placed after the opening triple-quote became the **first
line of the Lua source**. `#` is not a Lua comment, so Redis would have
rejected the script — but only with a real Redis, and the unit tests exercise
the local-bucket fallback. **The whole suite was blind to it.** Found by
reading the diff; fixed, verified against real Redis (5 allowed / 3 blocked at
5 rpm), and four regression tests added.

### 5.5 Three Prometheus metrics were defined but never called

`cache_operations_total`, `model_load_total` and `inference_in_progress` were
declared and exported but never incremented, so a dashboard panel and an alert
rule would have been permanently blank. Found by querying Prometheus after a
live run rather than trusting the code. Now wired and verified.

### 5.6 The API image did not contain the worker package

`batch.py` imported `worker.tasks` to enqueue jobs, but the API image
deliberately omits worker code — so every batch submission returned 503.
Caught by an end-to-end run against the real stack. Fixed by enqueuing via
`send_task` (by name), which is also the better decoupling.

### 5.7 Validation accepted truncated images

The validator reads only the image header, on purpose, so that a
decompression bomb is caught before its pixels are allocated. The cost was
that a truncated file passed. Caught by a test; fixed by adding Pillow's
`verify()` (a CRC check that does not decode pixels).

### 5.8 Others, briefly

* `/batch/{job_id}` returned 500 instead of 503 when Redis was down.
* A typo'd model name silently fell back to the default model instead of
  returning 404 — a client error was being hidden.
* Out-of-range detection scores crashed with a 500; now clamped and logged.
* The Nginx healthcheck failed because `localhost` resolves to IPv6 first
  while the server listened only on IPv4.
* SQLite rejected PostgreSQL pool arguments, silently skipping 18 database
  tests.

---

## 6. If there were more time

In priority order:

1. **Run the full Tiny-ImageNet fine-tune on a GPU** and register the result.
2. **Execute and benchmark the TensorRT path** on real hardware.
3. **Move the similarity index to a shared store** (pgvector or FAISS on a
   shared volume) so it survives horizontal scaling.
4. **Measure accuracy properly** against ImageNet and COCO validation sets.
5. **Commit an initial Alembic migration**.
6. **Calibrate confidence** (temperature scaling) — ECE of 0.22 is poor, and
   the API currently exposes uncalibrated scores.
7. **Add OpenTelemetry tracing** — correlation IDs give a trail through the
   logs; spans would give it through the timing too.
