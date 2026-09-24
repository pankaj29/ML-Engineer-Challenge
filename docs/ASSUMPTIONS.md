# Assumptions and Design Decisions

The brief says: *"Feel free to make reasonable assumptions and document them."*
This is that document.

It covers three things:

1. **[Ambiguities in the brief](#1-ambiguities-in-the-brief)** — where the
   requirements could be read more than one way, and which reading I took.
2. **[What is delivered](#2-what-is-delivered)** — each deliverable with the
   measurements behind it, and the limits stated plainly.
3. **[Engineering decisions](#3-engineering-decisions)** — choices I made where
   the brief was silent, and why.

Then two shorter sections: [deviations from the provided
scaffolding](#4-deviations-from-the-provided-scaffolding), and [what I would do
next](#5-what-i-would-do-next).

---

## 1. Ambiguities in the brief

### 1.1 "Utilise 3 different models", but only two are listed

The overview describes a system serving *"image classification, object
detection, and image similarity search"*. The numbered requirement says
**"Utilise 3 different models"** and then lists two bullets. The prescribed API
tree has only `classification.py` and `detection.py`.

I took the third model to be **image similarity search**, following the
overview. I confirmed this with the requester before building.

Delivered: `resnet50` (classification), `yolov8n` (detection),
`resnet50-embed` (similarity), plus `resnet50-tiny-imagenet` from the
fine-tuning requirement. The similarity endpoints live under
`/api/v1/similarity/*`.

### 1.2 CIFAR-100 or Tiny-ImageNet?

The brief names both datasets in consecutive bullets:

> - Image Classification (ViT / ResNet / EfficientNet etc. **on CIFAR-100**)
> - **For Image classification only**: fine-tune the model with ... mixed
>   precision, gradient clipping, learning rate scheduling. Use the
>   **tiny-ImageNet dataset** for this.

These are different datasets. CIFAR-100 is 32×32 with 100 classes;
Tiny-ImageNet is 64×64 with 200. They need different input handling and give
different models.

I treated **Tiny-ImageNet as the operative instruction**, for three reasons.
It is the explicit one, and it is attached to the fine-tuning requirement that
carries the three techniques actually being assessed — the CIFAR-100 mention
sits in a parenthetical alongside a list of candidate architectures, which
reads as an example rather than a mandate. It is also the only dataset the
brief gives a download command for, and the one its Getting Started section
fetches. And it is the harder task.

CIFAR-100 is not ruled out. `scripts/download_datasets.py` already supports it
(`--dataset cifar100`), and the training pipeline is dataset-shaped rather than
Tiny-ImageNet-specific: adding it means one `Dataset` class alongside
`models/training/dataset.py` and different normalisation constants.

### 1.3 Two paths given for the dataset script

The brief gives `./scripts/download_datasets.py` in Part 1 and
`scripts/setup/download_datasets.py` in Getting Started. The file is at the
first path. I read this as one script documented inconsistently, left it where
it is, and used the working path in the README.

### 1.4 The fine-tuned model is not the model the API serves by default

Part 1 asks for a classifier fine-tuned on Tiny-ImageNet. Part 2 asks for a
production classification API. Those pull in different directions: a
Tiny-ImageNet model knows 200 classes of 64×64 thumbnails, which is not what
you would put behind a general-purpose classification endpoint.

I built both and kept them separate. The fine-tuning pipeline is complete and
its output is registered and servable as `resnet50-tiny-imagenet`. The
**default** classification model is ImageNet-1k ResNet-50, because that is what
a real classification API would serve. Callers can select either per request.

Making the fine-tuned model the default is a one-line registry change.

### 1.5 "COCO subset" for detection

I took pretrained COCO weights to satisfy this. They are trained on COCO, which
contains any subset of it, and training a detector from scratch on a subset
would produce a worse model for considerably more effort. The `coco_sample`
download (val2017) is available in the provided script if a subset is wanted
for evaluation.

### 1.6 Grafana is marked "optional but preferred"

Delivered, with the data source provisioned automatically and a 22-panel
dashboard. Verified running.

---

## 2. What is delivered

### 2.1 The fine-tuned classifier

**Requirement:** fine-tune on Tiny-ImageNet with mixed precision, gradient
clipping and learning rate scheduling.

All 200 classes and all 100,000 training images, with no subsetting —
`verify_full_dataset()` refuses to start otherwise.

| | |
| --- | --- |
| Architecture | ResNet-50, ImageNet-1k weights, original stem |
| Hardware | NVIDIA A100-SXM4-40GB |
| Input | 224×224 |
| Epochs | 60, cosine schedule with 5% linear warmup |
| Batch size | 256 |
| Optimiser | AdamW, lr 3e-4, weight decay 5e-2 |
| Regularisation | label smoothing 0.1, RandAugment, MixUp, CutMix, RandomErasing |
| Weight averaging | EMA, decay 0.9998 |
| Time per epoch | ~95 s |
| **Top-1** | **78.91%** |
| **Top-5** | **92.12%** |
| Random baseline | 0.5% top-1 |

The best score came at epoch 57 from the EMA weights. All three required
techniques are active: `torch.autocast` fp16 with `GradScaler`,
`clip_grad_norm_` applied after unscaling, and cosine LR with warmup.

**Checked independently of the training loop.** `models/validation/validate.py`
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
| latency | p50 25.6 ms, p95 33.8 ms (CPU) |

The 78.60% here against 78.91% from training is the 2000-sample subset versus
the full 10,000-image validation set. That is ordinary sampling variance, in
the direction and size you would expect.

**On input resolution.** Tiny-ImageNet images are natively 64×64, so any
larger input is an upsample. I measured three configurations:

| Input | Stem | Top-1 | Time/epoch |
| --- | --- | ---: | ---: |
| 64×64 | adapted for small images | 73.98% | 82 s |
| 128×128 | original ImageNet | 77.66% | 38 s |
| **224×224** | original ImageNet | **78.91%** | 95 s |

64×64 looks like the natural choice and is the worst of the three, because
ResNet-50's stem downsamples 4× and has to be replaced at that resolution,
which throws away pretrained weights and makes every later layer run at four
times the spatial area. It is both less accurate and slower per epoch.

Going from 128 to 224 buys 1.25 points for 2.5× the compute. That is a much
smaller return than a resolution change usually gives, and the reason is that
upsampling adds no information — the source is still 64×64. The ceiling here
is the dataset, not the input size. I would not expect much more from this
architecture on this data; a stronger backbone is the lever that remains.

**Early stopping is off** (`--patience 0`). A cosine schedule does most of its
work in the final anneal, so validation accuracy plateaus mid-run as a matter
of course. Stopping on that plateau discards the part of the schedule that
pays. Early stopping is the right tool for `--scheduler plateau` or `step`;
with a fixed-length schedule it is a way to throw the schedule away.

**Serving preprocessing is tied to the model.** `TINY_IMAGENET_PREPROCESS` is
224×224, and `tests/unit/test_preprocessing_parity.py` derives its expected
size from that constant instead of hard-coding a number, so training and
serving cannot drift apart silently.

**Why not CPU.** Measured on the build machine (Intel Core Ultra 7 155H, 16
threads), timing real training steps rather than estimating:

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
produces a well-formed graph computing the wrong thing is worse than one that
fails loudly.

For the fine-tuned classifier:

| | |
| --- | --- |
| Size | 95.6 MB |
| Max absolute difference vs PyTorch | 3.46e-06 |
| Mean absolute difference | 3.69e-07 |
| Top-1 prediction agreement | identical |
| Dynamic batch | verified at batch 4 |

### 2.3 Quantization

INT8 is applied to all four models. For the fine-tuned classifier, static QDQ
calibrated on 200 real validation images:

| | fp32 | INT8 static |
| --- | ---: | ---: |
| Size | 95.6 MB | 24.4 MB (3.91× smaller) |
| CPU p50, batch 1 | 14.84 ms | 24.80 ms |
| Top-1 on 500 validation images | 76.80% | 65.60% |
| Agreement with fp32 | — | 71.20% |

**INT8 is registered but is not the default**, and those last two rows are why.
It is nearly four times smaller, but on this hardware it is also slower, and it
changes the answer on nearly three images in ten. That is a real trade, not a
free win. Callers who want the smaller model can ask for it per request.

The same pattern holds for the ImageNet ResNet-50, where I also measured
dynamic quantization:

| Variant | p50, batch 1 | Size |
| --- | ---: | ---: |
| ONNX fp32 | 75.7 ms | 97.4 MB |
| INT8 dynamic | 1008.0 ms | 24.5 MB |
| INT8 static QDQ | 104.6 ms | 24.9 MB |

Dynamic quantization was **13× slower than fp32**. It recomputes activation
scales on every call and falls back to poorly optimised integer convolution
kernels, which is a bad fit for convolutional networks. Static QDQ with real
calibration images is about ten times faster than dynamic, though still around
1.4× slower than fp32. Shipping a 13× slower "optimisation" as the default
because the brief said to apply quantization would have been the wrong call.

### 2.4 TensorRT

Both precisions built and benchmarked on the A100 with TensorRT 11.3:

| Precision | ONNX | Engine | Build | p50 | p95 | Throughput | Max diff vs fp32 ONNX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 91.2 MB | 91.5 MB | 23 s | 1.099 ms | 1.130 ms | 907 img/s | 3.02e-03 |
| fp16 | 45.6 MB | 46.0 MB | 28 s | **0.859 ms** | 0.905 ms | **1158 img/s** | 1.25e-02 |

fp16 is 1.28× faster at half the size.

**The fp16 engine does not pass the numerical check, and I have left that
reported rather than waved through.** `_verify_engine` compares against the
original fp32 ONNX with a 1e-2 tolerance; fp16 measured 1.25e-02, so
`verified=False`.

That tolerance is a poor test for fp16 and should be read as one. It bounds the
absolute difference of raw logits, which scale freely, and an fp16 mantissa
carries about three decimal digits — so 1e-2 on logits of order 10 is ordinary
rounding, not a defect. The question that matters for a classifier is whether
the predictions agree, and the validation run answers it: no predictions flip
under noise substantially larger than this. The right fix is to verify by top-1
agreement rather than logit distance. Until that is done the flag stays honest
and failing rather than being relaxed to make it pass.

**TensorRT INT8 is not built.** In the strongly-typed era an INT8 engine needs
a QDQ graph. `quantize.py` already produces one
(`<name>_int8_static.onnx`), so the path is short, but I have not run it.
Passing `precision="int8"` raises `UnsupportedPrecisionError` and names that
file. It would have been easy to set no flag, label the result INT8, and let
the engine build and the benchmark populate with numbers that were all wrong.

**Engine binaries are not committed.** A TensorRT engine is built for one
specific GPU architecture and TensorRT version and will not load on anything
else. The build metadata travels; the 140 MB of binaries do not.

### 2.5 Published accuracy for the pretrained models is cited, not re-measured

Top-1/top-5 for ImageNet ResNet-50 and mAP for YOLOv8n are the published
figures for those checkpoints. Re-measuring needs the ImageNet validation set
(~6 GB, account required) and COCO (~1 GB).

What I verified instead: that they produce correct predictions on real images
end to end, that ONNX export is numerically faithful to PyTorch, and that they
pass every behavioural validation check. The validation pipeline refuses to
report accuracy when the model's label space does not match the evaluation
set's, so it cannot produce a meaningless number by comparing a 1000-class
model against a 200-class dataset.

### 2.6 Known limits

**The similarity index lives in one process's memory.** With several API
replicas each has its own index. That is correct for a single instance and
wrong for a scaled deployment. Options are set out in `docs/TECHNICAL.md`.

**Alembic is configured but no migrations are committed.** Tables are created
with `Base.metadata.create_all` outside production. Production should use
versioned migrations; the dependency is present and the models are
migration-ready, but there is no initial revision.

**Confidence is not calibrated.** ECE is 0.1244, inside the 0.15 threshold the
validation pipeline enforces but not good. The API exposes raw softmax scores,
so a 0.9 does not mean 90% correct. Temperature scaling would fix it.

**ONNX Runtime fell back to CPU on the GPU box.** The benchmark run on the A100
had no CUDA execution provider available, so the ONNX numbers in
`benchmarks/reports/BENCHMARKS_GPU_CPU_FALLBACK.md` are CPU numbers. The file
is named that way deliberately: the benchmark script checks which device was
actually used and renames the report rather than publishing CPU timings under
a GPU filename. The TensorRT figures in §2.4 are the only true GPU
measurements in this repository.

---

## 3. Engineering decisions

### 3.1 Two files added to the prescribed API structure

The brief's tree is followed exactly, plus two files:

- `api/config.py` — required by "no hardcoded secrets" and "environment-based
  configuration".
- `api/dependencies.py` — shared FastAPI dependencies, so image extraction and
  validation are defined once rather than repeated in each router.

The extra routers (`batch.py`, `models.py`, `health.py`, `metrics.py`,
`similarity.py`) exist because the brief requires those endpoints.

### 3.2 Auth fails closed, the cache fails soft, the rate limiter fails open

Three deliberate and deliberately different choices:

- **Authentication fails closed.** If no API keys are configured, every request
  is rejected. There is no default credential.
- **The cache fails soft.** Every failure degrades to a cache miss, so the
  system gets slower, never wrong.
- **The rate limiter fails open.** If Redis is down, traffic is allowed, with a
  local per-process bucket as a partial backstop. A cache outage should not
  become a total outage.

The third is the one worth arguing about. Failing closed on the limiter would
turn a Redis blip into a full outage; failing open means a brief window where
limits are per-process rather than global. For this system that is the better
trade, and the local bucket keeps it from being unbounded.

### 3.3 Single images are synchronous, batches are not

Single-image endpoints respond directly. Batches return HTTP 202 and a job id,
because a batch can exceed any sensible HTTP timeout. A batch also costs N
rate-limit tokens rather than 1, so it cannot be used to slip past a tier
limit.

### 3.4 Images are never stored

Only a SHA-256 of the bytes, plus dimensions and format. That is enough for
caching, deduplication and drift monitoring, and it means nothing sensitive is
retained.

### 3.5 Tests run with no external services

Redis is replaced by `fakeredis`, PostgreSQL by in-memory SQLite, and models by
a `FakeRuntime`. A suite that needs infrastructure running is a suite that gets
skipped. Tests that need real artifacts exist separately and skip cleanly when
the artifacts are absent.

Unit coverage of `api/` is 89.6% across 734 tests. The brief's mandatory bar is
85% for critical paths; its stretch target is 90%.

### 3.6 Data lives inside the repository

Requested explicitly. Worth knowing: the repository sits inside a
OneDrive-synced folder, so 120k dataset files will sync. `data/` is gitignored,
and pausing OneDrive sync during dataset work is advisable.

### 3.7 Model artifacts are committed through Git LFS; the dataset is not

These look like one decision and are two.

**Artifacts go in** — 457 MB across 9 files. They cannot be reproduced without
a GPU session and several hours, so a reviewer cloning the repository would
otherwise have no way to run the model tests or serve a prediction.

**The dataset stays out** — 519 MB, 120,203 files. It *is* reproducible from
one documented command, so committing it would store half a gigabyte to save a
five-minute download.

I weighed three options:

| Option | Why not |
| --- | --- |
| Commit binaries directly | ~460 MB in history permanently, and `resnet50.onnx` at 97.4 MB is 2.6 MB under the size at which GitHub rejects a push outright. A slightly larger future export would break it. |
| GitHub Release assets | No history cost and 2 GB per file, but the artifacts stop being versioned alongside the code that expects them. |
| **Git LFS** (chosen) | Repository stays ~2 MB, artifacts are versioned with the code, and CI fetches them with `lfs: true`. |

The cost, stated plainly: GitHub's free tier gives 1 GB of LFS storage and 1 GB
of bandwidth per month. At 457 MB, storage is comfortable but roughly two full
clones a month exhausts the bandwidth. For a take-home submission that is the
right trade. For a busy repository it would not be, and Release assets would
win.

A clone made without git-lfs gets pointer files instead of models. Tests that
need a real artifact detect this and skip with `git lfs pull` as the stated
remedy, rather than failing with something unhelpful.

---

## 4. Deviations from the provided scaffolding

Three things in the starter code do not work as documented. All three are
fixed or worked around, and the changes are worth knowing about because two of
them change results rather than crashing.

### 4.1 `scripts/download_datasets.py` could not run the documented command

```python
choices=list(DatasetDownloader({}).datasets.keys()) + ["all"]
```

`{}` is passed where a path is expected, so `Path({})` raises `TypeError`
before any argument is parsed. The command in the brief's Getting Started
section fails immediately. Fixed by constructing the throwaway instance with
the real default directory.

### 4.2 `scripts/tiny_imagenet_dataloader.py` mislabels the validation set

```python
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), ...)
```

Tiny-ImageNet's validation split is not in `ImageFolder` layout. It is a flat
`val/images/` directory plus a `val_annotations.txt` mapping file.
`ImageFolder` treats `images` as a single class and assigns label 0 to all
10,000 validation images.

This is the dangerous one, because it does not crash. Validation accuracy reads
a meaningless ~0.5% and looks like a broken training loop. Confirmed against
the real dataset: `ImageFolder on val/ -> 10000 images, 1 distinct label:
{0: 10000}`.

The script now reads `val_annotations.txt` and reuses the training split's
class-to-index mapping so the two splits line up. Its public API is unchanged,
so any existing caller keeps working and simply gets correct labels. A
self-check in `__main__` fails loudly if the validation labels are ever
degenerate again. After the fix: 200 distinct labels, exactly 50 images each.

`models/training/dataset.py` has a fuller loader with readable class names, and
that is what the training pipeline uses.

### 4.3 `scripts/sample_serving_script.py` is a synchronous Flask stub

Single-threaded, no validation, no error handling, model in the request path. I
treated it as illustrative. The delivered API shares none of it.

---

## 5. What I would do next

In priority order:

1. **Calibrate confidence.** ECE 0.1244 means the scores the API returns are
   not probabilities. Temperature scaling is cheap and would make them usable
   for thresholding.
2. **Move the similarity index to a shared store** — pgvector, or FAISS on a
   shared volume — so it survives horizontal scaling.
3. **Verify fp16 TensorRT by prediction agreement** rather than logit distance,
   so the check measures something meaningful and the honest failure in §2.4
   goes away for the right reason.
4. **Build the TensorRT INT8 engine** from the QDQ graph that already exists.
5. **Try a stronger backbone.** Resolution is exhausted as a lever on this
   dataset; ConvNeXt or a ViT with ImageNet-21k weights is where the remaining
   accuracy is.
6. **Commit an initial Alembic migration.**
7. **Measure accuracy properly** for the pretrained models against the real
   ImageNet and COCO validation sets.
8. **Add OpenTelemetry tracing.** Correlation IDs give a trail through the
   logs; spans would give one through the timing too.
