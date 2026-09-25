# Assumptions and Design Decisions

The brief says to make reasonable assumptions and document them. This is that
document.

1. [Ambiguities in the brief](#1-ambiguities-in-the-brief)
2. [What is delivered](#2-what-is-delivered)
3. [Engineering decisions](#3-engineering-decisions)
4. [Deviations from the provided scaffolding](#4-deviations-from-the-provided-scaffolding)
5. [What I would do next](#5-what-i-would-do-next)

---

## 1. Ambiguities in the brief

### 1.1 Three models, but only two listed

The overview describes classification, detection and similarity search. The
numbered requirement says "utilise 3 different models" and then lists two. The
prescribed API tree has only `classification.py` and `detection.py`.

I took the third to be similarity search, following the overview, and
confirmed that with the requester before building.

Delivered: `resnet50` for classification, `yolov8n` for detection,
`resnet50-embed` for similarity, and `resnet50-tiny-imagenet` from the
fine-tuning requirement. Similarity endpoints live under
`/api/v1/similarity/*`.

### 1.2 CIFAR-100 or Tiny-ImageNet?

The brief names both, in consecutive bullets:

> - Image Classification (ViT / ResNet / EfficientNet etc. on CIFAR-100)
> - For Image classification only: fine-tune the model with ... mixed
>   precision, gradient clipping, learning rate scheduling. Use the
>   tiny-ImageNet dataset for this.

These are different datasets. CIFAR-100 is 32×32 with 100 classes,
Tiny-ImageNet is 64×64 with 200.

I went with Tiny-ImageNet. It is the explicit instruction, and it is attached
to the fine-tuning requirement that carries the three techniques being
assessed. The CIFAR-100 mention sits in a parenthetical next to a list of
candidate architectures, which reads as an example. Tiny-ImageNet is also the
only dataset the brief gives a download command for.

CIFAR-100 is not ruled out. `scripts/download_datasets.py` supports it
already, and the training pipeline is not Tiny-ImageNet-specific: adding it
means one `Dataset` class and different normalisation constants.

### 1.3 Two paths given for the dataset script

Part 1 says `./scripts/download_datasets.py`, Getting Started says
`scripts/setup/download_datasets.py`. The file is at the first path. I left it
there and used the working path in the README.

### 1.4 The fine-tuned model is not the default

Part 1 asks for a classifier fine-tuned on Tiny-ImageNet. Part 2 asks for a
production classification API. A Tiny-ImageNet model knows 200 classes of
64×64 thumbnails, which is not what you would put behind a general-purpose
endpoint.

I built both. The fine-tuned model is registered and servable as
`resnet50-tiny-imagenet`. The default for `/api/v1/classify` is ImageNet-1k
ResNet-50, because that is what a real classification API would serve.
Callers pick either per request, and making the fine-tuned one the default is
a one-line registry change.

### 1.5 "COCO subset" for detection

Pretrained COCO weights satisfy this. They are trained on COCO, which contains
any subset of it, and training a detector from scratch on a subset would give
a worse model for much more effort. The `coco_sample` download (val2017) is
available in the provided script if a subset is wanted for evaluation.

### 1.6 Grafana, marked optional

Delivered, with the data source provisioned automatically and a 22-panel
dashboard.

---

## 2. What is delivered

### 2.1 The fine-tuned classifier

Fine-tuned on Tiny-ImageNet with mixed precision, gradient clipping and
learning rate scheduling, over all 200 classes and all 100,000 training
images. `verify_full_dataset()` refuses to start on a partial dataset.

| | |
| --- | --- |
| Architecture | ResNet-50, ImageNet-1k weights, original stem |
| Hardware | NVIDIA A100-SXM4-40GB |
| Input | 224×224 |
| Epochs | 60, cosine with 5% linear warmup |
| Batch size | 256 |
| Optimiser | AdamW, lr 3e-4, weight decay 5e-2 |
| Regularisation | label smoothing 0.1, RandAugment, MixUp, CutMix, RandomErasing |
| Weight averaging | EMA, decay 0.9998 |
| Time per epoch | ~95 s |
| **Top-1** | **78.91%** |
| **Top-5** | **92.12%** |
| Random baseline | 0.5% |

Best score came at epoch 57, from the EMA weights. All three required
techniques are active: `torch.autocast` fp16 with `GradScaler`,
`clip_grad_norm_` after unscaling, and cosine LR with warmup.

Checked separately from the training loop. `models/validation/validate.py`
runs against the exported ONNX through the same `ModelService` the API uses,
on 2000 held-out images. Nine of nine checks pass:

| Check | Result |
| --- | --- |
| accuracy | top-1 78.60%, top-5 91.95% |
| calibration | ECE 0.1244 (threshold 0.15) |
| inference errors | 0 of 2000 samples failed |
| determinism | max diff 0.00e+00 across 3 runs |
| batch invariance | max diff 0.00e+00 |
| robustness | 0.0% of predictions flip under σ=0.01 noise |
| output sanity | no NaN or infinite values |
| artifact integrity | all artifacts present |
| latency | p50 25.6 ms, p95 33.8 ms on CPU |

78.60% against 78.91% is the 2000-sample subset versus the full 10,000-image
validation set. Ordinary sampling variance.

#### Input resolution

Tiny-ImageNet images are 64×64, so any larger input is an upsample. I measured
three configurations:

| Input | Stem | Top-1 | Time/epoch |
| --- | --- | ---: | ---: |
| 64×64 | adapted for small images | 73.98% | 82 s |
| 128×128 | original ImageNet | 77.66% | 38 s |
| **224×224** | original ImageNet | **78.91%** | 95 s |

The native resolution is the worst of the three. ResNet-50's stem downsamples
4×, so at 64px the stem has to be replaced, which throws away pretrained
weights and makes every later layer run at four times the spatial area. It is
less accurate and slower per epoch.

Going from 128 to 224 buys 1.25 points for 2.5× the compute. That is a small
return for a resolution change, and the reason is that upsampling adds no
information: the source is still 64×64. The ceiling is the dataset, not the
input size. A stronger backbone is the lever that remains.

#### Early stopping is off

`--patience 0`. A cosine schedule does most of its work in the final anneal,
so validation accuracy plateaus mid-run as a matter of course, and stopping on
that plateau discards the part of the schedule that pays. Early stopping suits
`--scheduler plateau` or `step`. With a fixed-length schedule it throws the
schedule away.

#### Preprocessing is tied to the model

`TINY_IMAGENET_PREPROCESS` is 224×224, and
`tests/unit/test_preprocessing_parity.py` reads its expected size from that
constant rather than hard-coding a number, so training and serving cannot
drift apart quietly.

#### Why not CPU

Measured on the build machine (Intel Core Ultra 7 155H, 16 threads), timing
real training steps:

| Config | img/s | 1 epoch | 30 epochs |
| --- | ---: | ---: | ---: |
| resnet50 + 64px stem | 3.8 | 7.6 h | 9.4 days |
| resnet50, original stem | 21.2 | 1.4 h | 1.7 days |
| resnet18 + 64px stem | 9.2 | 3.1 h | 3.9 days |
| resnet18, original stem | 75.0 | 23 min | 11.5 h |

The run that produced the numbers above:

```bash
python -m models.training.train_classifier \
    --arch resnet50 --epochs 60 --image-size 224 --no-stem-adapt \
    --batch-size 256 --lr 3e-4 --scheduler cosine --warmup-ratio 0.05 \
    --grad-clip 1.0 --label-smoothing 0.1 --patience 0 --ema \
    --device cuda
```

### 2.2 ONNX export

Every model is exported to ONNX and checked numerically against the PyTorch
graph it came from, not just checked for structural validity. An export that
produces a valid graph computing the wrong thing is worse than one that fails
loudly.

For the fine-tuned classifier:

| | |
| --- | --- |
| Size | 95.6 MB |
| Max absolute difference vs PyTorch | 3.46e-06 |
| Mean absolute difference | 3.69e-07 |
| Top-1 prediction | identical |
| Dynamic batch | verified at batch 4 |

### 2.3 Quantization

INT8 applied to all four models. For the fine-tuned classifier, static QDQ
calibrated on 200 real validation images:

| | fp32 | INT8 static |
| --- | ---: | ---: |
| Size | 95.6 MB | 24.4 MB (3.91× smaller) |
| CPU p50, batch 1 | 14.84 ms | 24.80 ms |
| Top-1 on 500 validation images | 76.80% | 65.60% |
| Agreement with fp32 | — | 71.20% |

INT8 is registered but is not the default. It is nearly four times smaller,
but on this hardware it is also slower, and it changes the answer on almost
three images in ten. Callers who want the smaller model can ask for it per
request.

The same pattern holds for the ImageNet ResNet-50, where I also measured
dynamic quantization:

| Variant | p50, batch 1 | Size |
| --- | ---: | ---: |
| ONNX fp32 | 75.7 ms | 97.4 MB |
| INT8 dynamic | 1008.0 ms | 24.5 MB |
| INT8 static QDQ | 104.6 ms | 24.9 MB |

Dynamic quantization was 13× slower than fp32. It recomputes activation scales
on every call and falls back to poorly optimised integer convolution kernels,
which suits convolutional networks badly. Static QDQ with real calibration
images is about ten times faster than dynamic, though still around 1.4× slower
than fp32. Shipping a 13× slower optimisation as the default because the brief
said to apply quantization would have been the wrong call.

### 2.4 TensorRT

All three precisions built, verified and benchmarked on an A100-SXM4-40GB with
TensorRT 11.3.0.99, batch 1:

| Precision | ONNX | Engine | Build | p50 | p95 | Throughput | Max diff | Verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| fp32 | 91.2 MB | 91.5 MB | 24 s | 1.298 ms | 1.336 ms | 822 img/s | 2.62e-03 | yes |
| fp16 | 45.6 MB | 46.0 MB | 29 s | 0.990 ms | 1.008 ms | 1066 img/s | 2.19e-02 | yes |
| int8 | 23.1 MB | 24.1 MB | 24 s | 0.920 ms | 1.017 ms | 1068 img/s | 1.10e-01 | yes |

INT8 is 1.41x faster than fp32 and a quarter of its size.

INT8 and fp16 run at the same speed here. Across four runs they traded places
between 0.87 and 1.00 ms, which is contention on a shared A100 rather than a
real difference. At batch 1 this model is bound by memory traffic and kernel
launch overhead, not arithmetic, so halving the precision of the arithmetic
buys nothing. What INT8 does buy is half the engine. Anyone reading this table
for a throughput service should benchmark at their own batch size, where the
tensor cores become the bottleneck and INT8 should pull ahead.

#### What it took to build INT8

Four independent constraints, found one at a time, three of them only on real
hardware:

1. No INT32 into DequantizeLinear. ONNX Runtime quantizes biases to INT32,
   which is correct - a bias scale is `input_scale * weight_scale` and int8
   would overflow - but TensorRT's `DequantizeLinear` accepts only 8- and
   4-bit types. It fails at the first bias node. Fixed with
   `QuantizeBias: False`, which leaves biases in fp32.
2. Symmetric quantization only. Every zero point must be zero, and MinMax
   calibration fits each activation's true, lopsided range. Fixed with
   `ActivationSymmetric` and `WeightSymmetric`.
3. MinMax collapses once symmetric. This one produced no error at all. It
   built a valid engine that was useless. Symmetric makes the range
   `[-max|x|, +max|x|]`, so a post-ReLU activation, never negative, wastes half
   its 256 levels, and one outlier stretches the rest. Measured on 200 held-out
   validation images, calibrated on a disjoint 200:

   | Calibration | Top-1 agreement with fp32 | TensorRT |
   | --- | ---: | --- |
   | MinMax, asymmetric | 70.0% | rejects the graph |
   | MinMax, symmetric | 18.0% | accepts |
   | Entropy, symmetric | 18.0% | accepts, 4.6x slower to calibrate |
   | Percentile, symmetric | 95.0% | accepts |

   Percentile clips at 99.999% rather than at the single most extreme
   activation seen. It ends up more faithful than the asymmetric MinMax
   graph it replaces. The other three constraints stop the build. This one
   ships.
4. INT8 convolutions need input channels divisible by 4. A hardware kernel
   limit, so no amount of checking the file catches it. ResNet's stem conv
   takes 3 channels (RGB) and has no INT8 tactic, so the build dies at kernel
   selection with *"Could not find any implementation for node ... /conv1/Conv
   ..."*. `quantize.py` finds such convolutions by weight shape and leaves them
   in fp32 - one node here, 52 of 53 convolutions still quantized.

The first two are properties of the file, so `check_trt_qdq_graph()` now
reports both at once before any GPU work starts, rather than letting the parser
name whichever node it reaches first.

The INT8 graph is a separate artifact, `<name>_int8_trt.onnx`, not a
replacement for `<name>_int8_static.onnx`. The CPU INT8 figures in §2.3 were
measured against the latter, and quietly changing what that filename contains
would have invalidated them.

#### Verification

`_verify_engine` compares the engine against the fp32 ONNX graph and bounds the
difference as a fraction of the reference's peak magnitude, not as an
absolute number. Logit scale is a property of the model - this classifier spans
about ±6.7 - so an absolute bound means something different on every model.
Limits are 0.1% for fp32, 1% for fp16 and 10% for INT8; measured 0.05%, 0.40%
and 1.96%.

fp32 is not bit-exact because TensorRT defaults to TF32 for fp32 matmuls on
Ampere, keeping 10 mantissa bits against fp32's 23.

The comparison runs on a real photograph from `samples/`, not random noise.
Noise broke the check in both directions at once: it produces smaller logits
(peak 2.63 against 5.61) and larger quantization error (0.710 against 0.125),
because the INT8 ranges were calibrated on photographs and noise falls outside
all of them. The ratio came out 27% against the real image's 2.2%, and since
the noise was redrawn each run the INT8 verdict flipped between builds of the
same engine. The input is now a fixed image, with a seeded fallback.

#### Engine binaries are not committed

A TensorRT engine is built for one GPU architecture and TensorRT version and
will not load on anything else. The build metadata travels as
`<name>.<precision>.json`; the 165 MB of binaries do not.

### 2.5 Published accuracy is cited, not re-measured

Top-1/top-5 for ImageNet ResNet-50 and mAP for YOLOv8n are the published
figures for those checkpoints. Re-measuring needs the ImageNet validation set
(~6 GB, account required) and COCO (~1 GB).

What I did verify: they produce correct predictions on real images end to end,
ONNX export is numerically faithful to PyTorch, and they pass every
behavioural validation check. The validation pipeline refuses to report
accuracy when the model's label space does not match the evaluation set's, so
it cannot produce a meaningless number by comparing a 1000-class model against
a 200-class dataset.

### 2.6 Known limits

The similarity index lives in one process's memory. With several API replicas
each has its own index, which is correct for a single instance and wrong for a
scaled deployment. Options are in `docs/TECHNICAL.md`.

Schema changes go through Alembic. An initial revision is committed, and a
`migrate` init container runs `alembic upgrade head` before the API starts.
Outside production `Base.metadata.create_all` still handles it, which keeps a
local run from needing a migration step.

This was not always true, and the gap was worse than it sounds. Production
sets `create_tables=False`, so with no migrations the inference log simply did
not exist. Writes to it are swallowed by design, so nothing errored: drift
detection read an empty table, reported no drift, and the retraining loop
agreed. It was found by deploying to a cluster and looking for the rows.

`env.py` excludes `similarity_vectors` from autogenerate. That table is
created by `pgvector_index.py` rather than the ORM, because its column width
comes from the embedding model, so autogenerate sees a table with no model
behind it and writes a `DROP`. Without the exclusion the first migration after
any schema change would delete the similarity index.

Confidence is not calibrated. ECE is 0.1244, inside the 0.15 threshold the
validation pipeline enforces but not good. The API returns raw softmax scores,
so 0.9 does not mean 90% correct. Temperature scaling would fix it.

ONNX Runtime fell back to CPU on the GPU box. The benchmark run on the A100 had
no CUDA execution provider available, so the ONNX numbers in
`benchmarks/reports/BENCHMARKS_GPU_CPU_FALLBACK.md` are CPU numbers. The file
is named that way on purpose: the benchmark script checks which device was
actually used and renames the report rather than publishing CPU timings under
a GPU filename. The TensorRT figures in §2.4 are the only true GPU
measurements here.

---

## 3. Engineering decisions

### 3.1 Two files added to the prescribed structure

The brief's tree is followed exactly, plus:

- `api/config.py`, required by "no hardcoded secrets" and "environment-based
  configuration".
- `api/dependencies.py`, so image extraction and validation are defined once
  rather than repeated in each router.

The extra routers (`batch.py`, `models.py`, `health.py`, `metrics.py`,
`similarity.py`) exist because the brief requires those endpoints.

### 3.2 Auth fails closed, the cache fails soft, the rate limiter fails open

Three deliberately different choices:

- Authentication fails closed. With no API keys configured, every request is
  rejected. There is no default credential.
- The cache fails soft. Every failure becomes a cache miss, so the system gets
  slower, never wrong.
- The rate limiter fails open. If Redis is down traffic is allowed, with a
  per-process bucket as a partial backstop.

The third is the one worth arguing about. Failing closed would turn a Redis
blip into a full outage; failing open means a brief window where limits are
per-process rather than global. For this system that is the better trade, and
the local bucket keeps it bounded.

### 3.3 Single images are synchronous, batches are not

Single-image endpoints respond directly. Batches return HTTP 202 and a job id,
because a batch can exceed any sensible HTTP timeout. A batch also costs N
rate-limit tokens rather than 1, so it cannot be used to slip past a tier
limit.

### 3.4 Images are never stored

Only a SHA-256 of the bytes, plus dimensions and format. Enough for caching,
deduplication and drift monitoring, and nothing sensitive is retained.

### 3.5 Tests run with no external services

Redis is replaced by `fakeredis`, PostgreSQL by in-memory SQLite, and models by
a `FakeRuntime`. A suite that needs infrastructure running is a suite that gets
skipped.

Integration tests go the other way and use the real thing: real ONNX
artifacts, and real PostgreSQL and Redis when they are reachable. End-to-end
tests go further still and drive the deployed stack over HTTP, which is the
only level that sees nginx, the container image and the Celery worker. A
container serving an artifact the host replaced looks healthy to every
in-process test; only a request through the gateway notices. They skip
cleanly when those are absent. That split matters because the substitutes hide
real differences. SQLite has no native boolean, which is why `inference_stats`
sums a `case()` expression rather than casting, and only PostgreSQL can
confirm that workaround is right. The rate limiter's Lua script exists for
atomicity, and only a real server can show 20 concurrent requests against a
15-token bucket letting exactly 15 through.

Unit coverage of `api/` is 95.8%, with the cache, database and rate-limiting
services at 100%. The brief's mandatory bar is 85% on critical paths and its
target is 90%. CI gates at 92, low enough that a failed Git LFS fetch skipping
the artifact tests does not read as a coverage regression.

### 3.6 Data lives inside the repository

Requested explicitly. Worth knowing: the repository sits inside a
OneDrive-synced folder, so 120k dataset files will sync. `data/` is gitignored,
and pausing OneDrive sync during dataset work is advisable.

### 3.7 Model artifacts use Git LFS; the dataset does not

These look like one decision and are two.

Artifacts go in, 457 MB across 9 files. They cannot be reproduced without a GPU
session and several hours, so a reviewer cloning the repository would have no
way to run the model tests or serve a prediction.

The dataset stays out, 519 MB across 120,203 files. It is reproducible from one
documented command, so committing it would store half a gigabyte to save a
five-minute download.

Three options considered:

| Option | Why not |
| --- | --- |
| Commit binaries directly | ~460 MB in history permanently, and `resnet50.onnx` at 97.4 MB is 2.6 MB under the size at which GitHub rejects a push. A slightly larger future export would break it. |
| GitHub Release assets | No history cost and 2 GB per file, but the artifacts stop being versioned alongside the code that expects them. |
| **Git LFS** (chosen) | Repository stays ~2 MB, artifacts are versioned with the code, and CI fetches them with `lfs: true`. |

The cost: GitHub's free tier gives 1 GB of LFS storage and 1 GB of bandwidth a
month. At 457 MB storage is comfortable, but roughly two full clones a month
exhausts the bandwidth. For a take-home submission that is the right trade. For
a busy repository it would not be, and Release assets would win.

A clone made without git-lfs gets pointer files instead of models. Tests that
need a real artifact detect this and skip with `git lfs pull` as the remedy,
rather than failing with something unhelpful.

### 3.8 Replacing weights invalidates the cache automatically

Cache keys carry a content hash of the model artifact alongside its name and
version. Registering a new version is still the right practice, but if someone
overwrites weights in place the old cached predictions are simply never read
again. Without that, the cache would go on serving answers from a file that is
no longer on disk until the TTL expired, with no error and no latency change to
hint at it.

---

## 4. Deviations from the provided scaffolding

Three things in the starter code do not work as documented. Two of them change
results rather than crashing, which is why they are worth spelling out.

### 4.1 `download_datasets.py` cannot run the documented command

```python
choices=list(DatasetDownloader({}).datasets.keys()) + ["all"]
```

`{}` is passed where a path is expected, so `Path({})` raises `TypeError`
before any argument is parsed. The command in the brief's Getting Started
section fails immediately. Fixed by constructing the throwaway instance with
the real default directory.

### 4.2 `tiny_imagenet_dataloader.py` mislabels the validation set

```python
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), ...)
```

Tiny-ImageNet's validation split is not in `ImageFolder` layout. It is a flat
`val/images/` directory plus a `val_annotations.txt` mapping file.
`ImageFolder` treats `images` as a single class and gives label 0 to all 10,000
validation images.

This is the dangerous one, because it does not crash. Validation accuracy reads
a meaningless ~0.5% and looks like a broken training loop. Confirmed against
the real dataset: `ImageFolder on val/ -> 10000 images, 1 distinct label:
{0: 10000}`.

The script now reads `val_annotations.txt` and reuses the training split's
class-to-index mapping so the two line up. Its public API is unchanged, so any
existing caller keeps working and just gets correct labels. A self-check in
`__main__` fails loudly if the validation labels are ever degenerate again.
After the fix: 200 distinct labels, exactly 50 images each.

`models/training/dataset.py` has a fuller loader with readable class names, and
that is what the training pipeline uses.

### 4.3 `sample_serving_script.py` is a synchronous Flask stub

Single-threaded, no validation, no error handling, model in the request path. I
treated it as illustrative. The delivered API shares none of it.

---

## 5. What I would do next

1. Calibrate confidence. ECE 0.1244 means the scores the API returns are not
   probabilities. Temperature scaling is cheap and would make them usable for
   thresholding.
2. Move the similarity index to a shared store, pgvector or FAISS on a shared
   volume, so it survives horizontal scaling.
3. Benchmark TensorRT at larger batch sizes. At batch 1 INT8 and fp16 are
   indistinguishable; the tensor cores only become the bottleneck with more
   work in flight, which is where INT8 should separate.
4. Try a stronger backbone. Resolution is exhausted as a lever on this dataset;
   ConvNeXt or a ViT with ImageNet-21k weights is where the remaining accuracy
   is.
5. Measure accuracy properly for the pretrained models against the real
   ImageNet and COCO validation sets.
6. Add OpenTelemetry tracing. Correlation IDs give a trail through the logs;
   spans would give one through the timing too.
