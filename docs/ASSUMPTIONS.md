# Assumptions, Decisions and Known Gaps

The brief says: *"Feel free to make reasonable assumptions and document them."*
This is that document.

It is organised by how much it should matter to a reviewer:

1. **[Ambiguities in the brief](#1-ambiguities-in-the-brief)** — where the
   requirements were open to reading, and which reading was taken.
2. **[Completeness of each deliverable](#2-completeness-of-each-deliverable)** —
   what is finished with evidence, and what is not, stated plainly either way.
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

## 2. Completeness of each deliverable

This section used to be titled "Things not fully delivered". Two of its
entries - TensorRT and the fine-tuned classifier - have since been executed on
an A100 and are now recorded with measurements rather than intentions. The
remaining entries are still genuine gaps, and are marked as such.

### 2.1 TensorRT: now executed, with one precision unavailable

**Requirement:** *"Convert models to ONNX and TensorRT formats."*

**Status: done.** Both formats are produced and verified. This section used to
say the TensorRT code had never run; that is no longer true.

Executed on an NVIDIA A100-SXM4-40GB with **TensorRT 11.3**:

| Precision | ONNX | Engine | Build | p50 | p95 | Throughput | max diff vs fp32 ONNX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 91.2 MB | 91.5 MB | 24 s | 1.059 ms | 1.105 ms | 972 img/s | 1.62e-03 |
| fp16 | 45.6 MB | 46.0 MB | 31 s | **0.729 ms** | 0.749 ms | **1369 img/s** | 1.41e-02 |

fp16 is **1.45x faster and half the size**, which is the expected shape of the
result on an A100.

**The fp16 engine does not pass the numerical check, and that is reported
rather than waved through.** `_verify_engine` compares against the original
fp32 ONNX with a 1e-2 tolerance; fp16 measured 1.41e-02, so `verified=False`.

That tolerance is a poor test for fp16 and should be read as such. It bounds
the absolute difference of raw logits, which scale freely - an fp16 mantissa
carries about three decimal digits, so 1e-2 on logits of order 10 is ordinary
rounding, not a defect. The meaningful question for a classifier is whether
the *predictions* agree, and the evidence for that is the validation run: the
model scores 76.40% top-1 with 0.0% of predictions flipping under noise
substantially larger than this. A future improvement is to verify by top-1
agreement rather than logit distance; until then the flag is left honest and
failing rather than quietly relaxed to make it pass.

**The API had moved two generations underneath this code.** It was written
against TensorRT 10.x, the machine had 11.3, and three separate calls had been
removed in between. Each fix surfaced the next, which is worth recording
because the failure mode was identical every time - an `AttributeError` on a
symbol the documentation still describes:

| Removed | Era | Replacement used here |
| --- | --- | --- |
| `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` | gone in 10 | explicit batch is the only mode; pass no flag |
| `Builder.platform_has_fast_fp16` / `_int8` | gone in 10 | advisory only; skip the note when absent |
| `BuilderFlag.FP16` / `.INT8` | gone in 11 | networks are `STRONGLY_TYPED`; precision comes from the graph |

`export_tensorrt.py` now detects the era by **probing for attributes rather
than parsing a version string**, and supports all three. A version string
would have been the obvious approach and the wrong one: it encodes a guess
about which release removed what, which is precisely the thing that was wrong
four times over.

**fp16 needs an fp16 graph.** Because TensorRT 11 takes precision from the
ONNX dtypes, `convert_onnx_to_fp16()` rewrites the graph first
(`keep_io_types=True`, so inputs and outputs stay fp32 and callers are
unaffected). This uses `onnxruntime.transformers.float16`, already a
dependency. The obvious package, `onnxconverter-common`, is deliberately
avoided: it hard-pins `protobuf==3.20.2`, which drops protobuf below the
`>=6.31.1` that onnx requires and sends pip back to building onnx from source.

**INT8 via TensorRT is not done, and is refused rather than faked.** In the
strongly-typed era an INT8 engine needs a QDQ graph. `quantize.py` already
produces one (`<name>_int8_static.onnx`), so the path is short, but it has not
been run. Passing `precision="int8"` raises `UnsupportedPrecisionError` and
names that file. It would have been easy to set no flag and label the result
INT8 - the engine would build, the benchmark would populate, and every number
would be wrong.

**INT8 through ONNX Runtime is done** and is where the size reduction is
measured: 95.6 MB to 24.4 MB.

### 2.2 The fine-tuned classifier: trained, exported and validated

**Requirement:** fine-tune on Tiny-ImageNet with mixed precision, gradient
clipping and LR scheduling.

**Status: done.** All 200 classes, all 100,000 training images, no subsetting -
`verify_full_dataset()` refuses to start otherwise.

| | |
| --- | --- |
| Hardware | NVIDIA A100-SXM4-40GB |
| Epochs | 60 (cosine, 5% linear warmup) |
| Input | 128x128, original ImageNet stem |
| Batch size | 256 |
| Optimiser | AdamW, lr 3e-4, weight decay 5e-2 |
| Time per epoch | ~38 s |
| **Top-1** | **77.66%** |
| **Top-5** | **90.18%** |
| Random baseline | 0.5% top-1 |

All three required techniques were active: `torch.autocast` fp16 with
`GradScaler`, `clip_grad_norm_` after unscaling, and cosine LR with warmup.

**Independently validated**, not just reported by the training loop.
`models/validation/validate.py` against the exported ONNX, 2000 held-out
samples, 8 of 8 checks passing:

| Check | Result |
| --- | --- |
| accuracy | top-1 76.40%, top-5 90.70% |
| calibration | ECE 0.0632 (threshold 0.15) |
| determinism | max diff 0.00e+00 across 3 runs |
| batch invariance | max diff 0.00e+00 |
| robustness | 0.0% of predictions flip under sigma=0.01 noise |
| output sanity | no NaN or infinite values |
| artifact integrity | all artifacts present |
| latency | p95 7.1 ms |

The 76.40% here versus 77.66% from training is the 2000-sample subset versus
the full 10,000-image validation set - ordinary sampling variance, in the
direction and magnitude you would expect.

**Two findings from getting here, because the first attempt scored worse.**

*Resolution beats epochs.* An earlier 30-epoch run at 64x64 with the stem
adapted for small images reached 73.98%. Tiny-ImageNet is natively 64x64, so
that looks like the natural choice, but ResNet-50's stem downsamples 4x and
must be replaced at that resolution - discarding pretrained weights. Feeding
128x128 through the **original** stem gives `layer1` a 32x32 map and uses the
network as pretrained. It is also *cheaper* per epoch (38 s versus 82 s),
because 32x32 into `layer1` is a quarter the area of the adapted stem's 64x64.
Worth +3.68 points for less compute.

*The learning rate has to move with the stem.* Keeping lr 1e-3 after switching
to the original stem made validation accuracy **regress** - 70.5% to 64.8%
while training loss kept falling, with non-finite gradients appearing. 1e-3
suits a partly randomly-initialised network; with the whole pretrained model
intact it erodes the features the resolution change was meant to preserve.
3e-4 fixed it.

**Early stopping is off by default** (`--patience 0`), and that is deliberate.
It was set to 8 and terminated three runs at epoch 9. A cosine schedule does
most of its work in the final anneal, so a mid-run plateau is normal rather
than a signal to stop. The notebook sets 15 explicitly; anything much lower is
a trap with this schedule.

**Serving preprocessing moved with the model.** `TINY_IMAGENET_PREPROCESS` is
128x128, and `tests/unit/test_preprocessing_parity.py` derives its expected
size from that constant rather than hard-coding a number, so training and
serving cannot silently diverge again.

**Why CPU training was not an option.** Measured on the build machine
(Intel Core Ultra 7 155H, 16 threads), timing real training steps rather than
estimating:

| Config | img/s | 1 epoch | 30 epochs |
| --- | ---: | ---: | ---: |
| resnet50 + 64px stem | 3.8 | 7.6 h | **9.4 days** |
| resnet50, original stem | 21.2 | 1.4 h | 1.7 days |
| resnet18 + 64px stem | 9.2 | 3.1 h | 3.9 days |
| resnet18, original stem | 75.0 | 23 min | 11.5 h |

```bash
python -m models.training.train_classifier \
    --arch resnet50 --epochs 60 --image-size 128 --no-stem-adapt \
    --batch-size 256 --lr 3e-4 --scheduler cosine --warmup-ratio 0.05 \
    --grad-clip 1.0 --label-smoothing 0.1 --patience 15 --device cuda
```

**Subsetting has been removed.** The `--classes` and `--limit-batches` flags
used during development are gone, and `verify_full_dataset()` refuses to train
unless every class and every image found on disk was loaded.

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
