# Assumptions and Decisions

The brief asks for reasonable assumptions to be documented. These are they,
along with the decisions that shaped the delivery and the fixes made to the
scripts that came with the brief.

1. [Reading the brief](#1-reading-the-brief)
2. [Engineering decisions](#2-engineering-decisions)
3. [Fixes to the provided scripts](#3-fixes-to-the-provided-scripts)
4. [What is not done](#4-what-is-not-done)

---

## 1. Reading the brief

### 1.1 Three models, two listed

The overview describes classification, detection and similarity search; the
numbered requirement says "utilise 3 different models" and lists two, and the
prescribed tree has only `classification.py` and `detection.py`. I took the
third to be similarity search, following the overview, and confirmed it with
the requester before building. Delivered: `resnet50` (classification),
`yolov8n` (detection), `resnet50-embed` (similarity), plus
`resnet50-tiny-imagenet` from the fine-tuning requirement.

### 1.2 CIFAR-100 or Tiny-ImageNet

The brief names CIFAR-100 in a parenthetical next to candidate architectures,
then says to fine-tune "using the tiny-ImageNet dataset" in the bullet that
carries the three required techniques. I read the first as an example and the
second as the instruction; Tiny-ImageNet is also the only dataset the brief
gives a download command for. `scripts/download_datasets.py` supports
CIFAR-100 too, and adding it to training is one `Dataset` class.

### 1.3 Two paths for the dataset script

Part 1 says `./scripts/download_datasets.py`; Getting Started says
`scripts/setup/download_datasets.py`. The file is at the first path, and the
docs use that one.

### 1.4 The fine-tuned model is not the default

Part 1 asks for a classifier fine-tuned on Tiny-ImageNet, Part 2 for a
production classification API. 200 classes learned from 64x64 thumbnails is
not what you would put behind a general-purpose endpoint, so both exist: the
default for `/api/v1/classify` is ImageNet-1k ResNet-50, and the fine-tuned
model is one `"model_name"` away. Swapping the default is a one-line registry
change.

### 1.5 "COCO subset" for detection

The detector uses Ultralytics' COCO-trained weights; training from scratch on
a subset would give a worse model for far more effort. The subset is used for
what it is good for: evaluation. `models/validation/coco_eval.py` scores the
detector on 500 val2017 images through the serving path (0.392 mAP50-95 fp32,
0.381 INT8), and COCO images calibrate its INT8 model.

### 1.6 Grafana, marked optional

Delivered, with the Prometheus data source and an 18-panel dashboard
provisioned automatically.

---

## 2. Engineering decisions

### 2.1 Two files beyond the prescribed tree

The brief's `api/` layout is followed exactly, plus `api/config.py` (needed for
"environment-based configuration" and "no hardcoded secrets") and
`api/dependencies.py` (so image extraction is written once for every router).
The extra routers exist because the brief requires their endpoints.

### 2.2 Failure policy

Authentication fails closed: no keys configured means every request is
rejected. The cache fails soft: a failure is a miss. The rate limiter fails
open to per-process buckets and reconnects in the background, because turning
a Redis blip into a full outage is worse than a short window of per-replica
limits. API keys carry full access, including model reload; a JWT can be
issued with narrower scopes. A deployment that needs admin-only reloads should
issue scoped tokens and not hand out raw keys.

### 2.3 Single images are synchronous, batches are not

A batch can outlast any sensible HTTP timeout, so it returns 202 and a job id.
It costs one rate-limit token per image, so it cannot be used to slip past a
tier's limit, and only the key that submitted a job can read or cancel it.

### 2.4 Tests need nothing running

Unit tests use fakeredis, in-memory SQLite and a fake model runtime.
Integration tests go the other way and use real ONNX models and, when
reachable, real Redis, PostgreSQL and pgvector: SQLite has no native boolean
and only a real Redis proves the rate limiter's Lua script is atomic.
End-to-end tests drive the running stack through nginx.

Coverage is 94.5% on `api/` and 94.6% on `worker/`. The brief's
bar is 85% on critical paths with a 90% target, and the batch path lives in
`worker/`, so both are reported. `models/` is lower (75.9%) because it
holds the GPU training loop and CLIs that no request touches. CI gates `api/`
at 92%.

### 2.5 Model files in Git LFS, the dataset not

The ten model files (503 MB) cannot be reproduced without a GPU run, so they
are in Git LFS and CI fetches them. The dataset (240 MB, 120,203 files) is one
documented command away, so it stays out. Plain git was ruled out because
`resnet50.onnx` is 97.4 MiB, within 3 MiB of GitHub's 100 MiB per-file limit;
a slightly larger export would break every push. GitHub's free
LFS bandwidth is 1 GB a month, about two full clones; fine for a submission,
not for a busy repository.

### 2.6 Replaced weights never serve stale predictions

Cache keys include a content hash of the model file, so overwriting weights in
place changes every key. Without it the cache would keep serving answers from
a file no longer on disk until the TTL ran out, with no error.

### 2.7 The repository sits in a OneDrive folder

`data/` is git-ignored but OneDrive still syncs its 120,203 files. Pause sync
while working with the dataset.

---

## 3. Fixes to the provided scripts

Three things in the starter code did not work as documented. Two changed
results instead of crashing.

**`download_datasets.py` could not parse its arguments.** It built the choice
list from `DatasetDownloader({})`, and `Path({})` raises `TypeError` before any
argument is read, so the brief's own command failed. It now uses the real
default directory. A second fix: `tqdm` was imported at module level but lives
in the training requirements, so the script died in environments without it.

**`tiny_imagenet_dataloader.py` labelled every validation image 0.** The
validation split is a flat `val/images/` folder plus `val_annotations.txt`,
not `ImageFolder` layout, so `ImageFolder` saw one class. Validation accuracy
read about 0.5% and looked like a broken training loop. It now reads the
annotations and reuses the training split's class mapping: 200 labels, 50
images each. `models/training/dataset.py` is the fuller loader the training
pipeline uses.

**`sample_serving_script.py` is a synchronous Flask stub** with no validation
or error handling. It is illustrative; the delivered API shares none of it.

---

## 4. What is not done

- **ImageNet accuracy for the default classifier** is torchvision's published
  80.86%, not re-measured: the validation set needs an account.
- **TensorRT engines exist for the fine-tuned classifier only.** The builder is
  generic; the other three were not built on the rented A100.
- **Retrieval quality** of the embedding model is not measured; that needs a
  dataset with similarity labels.
- **Confidence is not calibrated.** The fine-tuned model is underconfident
  (ECE 0.128, mean confidence 0.661 against 0.789 accuracy). Temperature
  scaling is the fix.
- **Resolution experiments.** 224px was chosen over 64px and 128px from
  shorter development runs whose logs were not kept, so their numbers are not
  quoted anywhere.
- **Tracing.** Correlation ids connect the logs; OpenTelemetry spans would add
  timing across services.
