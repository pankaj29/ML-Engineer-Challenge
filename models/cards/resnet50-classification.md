# Model Card — ResNet-50 (Image Classification)

> A model card is an honest summary of what a model does, how well it does it,
> and — most importantly — where it fails. It exists so that whoever deploys
> the model knows what they are deploying, and whoever consumes its output
> knows how far to trust it.

| | |
| --- | --- |
| **Registry name** | `resnet50` |
| **Version** | `1.0.0` |
| **Task** | Image classification |
| **Status** | Active, default for the classification task |
| **Serving endpoint** | `POST /api/v1/classify` |
| **Licence** | Weights from torchvision (BSD-3-Clause); ImageNet terms apply to the training data |

---

## 1. What it does

Given one photograph, it returns a ranked list of the most likely object
categories, with a confidence score for each. It recognises **1,000
categories** — the ImageNet-1k label set, which covers everyday objects,
animals, vehicles, food and household items.

It answers "what is the main thing in this picture?". It does **not** say
where that thing is, count how many there are, or describe a scene. For
location and counting, use `POST /api/v1/detect` instead.

---

## 2. Architecture and why it was chosen

**ResNet-50**: a 50-layer residual convolutional network, ~25.6 M parameters.

The defining idea is the *residual connection*: each block learns a small
adjustment to its input rather than a whole new representation, and adds it
on. Before residual connections, networks past about 20 layers got *worse*
with depth because the training signal faded before reaching the early layers.
Residual connections give that signal a direct path, which is what made
50-layer networks trainable at all.

Why this rather than something newer:

* **Accuracy per millisecond on CPU.** A Vision Transformer of comparable
  accuracy needs roughly 3-4x the compute, and this system is CPU-first. At
  85 ms p50 for a single image, ResNet-50 leaves comfortable headroom under
  the project's 1-second requirement.
* **It exports cleanly.** Every operation has a well-supported ONNX
  equivalent, and it quantizes without special handling. Several more modern
  architectures need per-operator workarounds to export at all, which is a
  real cost when the pipeline must be reproducible.
* **It is thoroughly characterised.** Its failure modes are documented across
  a decade of literature, which is worth a lot when writing a section like
  section 6 of this card honestly.

**Trade-off accepted.** A ConvNeXt or an EfficientNetV2 would be 2-4 points
more accurate at a similar parameter count. That accuracy was traded for
latency headroom and export reliability. If accuracy became the binding
constraint, swapping the backbone is a one-line change in
`scripts/prepare_models.py` plus a new registry entry — the serving code is
architecture-agnostic.

---

## 3. Training data

The served weights are torchvision's `IMAGENET1K_V2` checkpoint, trained on
**ImageNet-1k**: ~1.28 M training images across 1,000 categories, scraped
from the web and labelled via crowdsourcing.

This project also fine-tunes ResNet-50 on **Tiny-ImageNet** (200 classes),
which the brief requires. That model has been trained in full — 77.66% top-1,
91.52% top-5 on an A100 — and is registered as `resnet50-tiny-imagenet`. It is
a separate model with its own card:
[`resnet50-tiny-imagenet.md`](resnet50-tiny-imagenet.md).

It is deliberately **not** the default here. A 200-class model built from 64x64
thumbnails is a poor general-purpose classification API next to ImageNet-1k at
224px — see `docs/ASSUMPTIONS.md` §1.4. Swapping the default is a one-line
registry change.

### Preprocessing (must match exactly)

| Step | Value |
| --- | --- |
| Resize | Shortest side to 256 (crop_pct 0.875) |
| Crop | Centre 224 x 224 |
| Colour | RGB, alpha composited onto white |
| Scale | 0-255 → 0-1 |
| Normalise | mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |

These constants are not decorative. They are the statistics the model was
trained with, and using different ones silently costs several points of
accuracy with no error message. They live in `api/utils/image_processing.py`
as `CLASSIFICATION_PREPROCESS` and travel with the model through the registry.

---

## 4. Measured performance

Measured on the build machine: **Intel Core Ultra 7 155H, 22 logical cores,
CPU only**, ONNX Runtime 1.26.0. Reproduce with
`python -m models.optimization.benchmark`.

### Latency

| Format | Batch | p50 | p95 | p99 | Throughput | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX fp32 | 1 | **84.8 ms** | 109.1 ms | 131.5 ms | 13.1 img/s | 97.4 MB |
| ONNX fp32 | 4 | 266.9 ms | 348.0 ms | 423.8 ms | 14.2 img/s | 97.4 MB |
| ONNX INT8 (static) | 1 | 120.3 ms | 202.8 ms | 274.4 ms | 7.2 img/s | **24.9 MB** |
| ONNX INT8 (static) | 4 | 523.8 ms | 666.3 ms | 788.9 ms | 8.1 img/s | 24.9 MB |

**Requirement check:** the brief asks for sub-second single-image inference.
p99 at batch 1 is 131 ms — roughly 7x inside budget.

**INT8 is deliberately not the default.** It is 3.92x smaller but ~1.4x
*slower* on this CPU. The reasoning, and the much worse result from dynamic
quantization, are in `docs/TECHNICAL.md`. INT8 is registered and selectable
per request (`"runtime": "onnx_int8"`) for memory-constrained deployments.

### Quantization fidelity

Static INT8, calibrated on 100 real images:

| Metric | Value |
| --- | --- |
| Size | 97.4 MB → 24.9 MB (3.92x) |
| Top-1 agreement with fp32 | **71.9%** |

That agreement number deserves a plain reading: on roughly 28 of every 100
images, the INT8 model picks a *different* top class from the fp32 model.
Many of those are near-ties between visually similar categories, but it is not
a change to make silently. **Do not switch a deployment to INT8 on size alone
without evaluating accuracy on your own data.**

### Accuracy

Reference accuracy for these weights on the ImageNet-1k validation set is
**80.86% top-1 / 95.43% top-5** (torchvision `IMAGENET1K_V2`).

**Not independently re-measured here.** Doing so requires the ImageNet
validation set, which needs a registered account and is ~6 GB. The validation
pipeline correctly *refuses* to report accuracy against Tiny-ImageNet, because
the two label spaces (1,000 vs 200 classes) are unrelated and comparing them
would produce a confident, meaningless number. See `docs/ASSUMPTIONS.md`.

---

## 5. Validation results

From `python -m models.validation.validate` — all checks pass:

| Check | Result |
| --- | --- |
| Artifact integrity | Pass — both artifacts present and non-empty |
| Determinism | Pass — identical output across 3 runs (max diff 0.0) |
| Batch invariance | Pass — a prediction does not depend on batch position |
| Output sanity | Pass — no NaN/Inf, probabilities sum to 1 |
| Robustness | Pass — 0% of predictions flip under imperceptible noise (σ=0.01) |
| Latency | Pass — p95 95.4 ms, well under the 1,000 ms budget |
| ONNX export fidelity | Pass — max abs diff vs PyTorch 2.4e-06 |

---

## 6. Limitations and failure modes

This is the section that matters most.

### It only knows 1,000 things

Anything outside the ImageNet label set will be confidently mapped to the
nearest category it *does* know. A photograph of a microscope slide, an X-ray,
a circuit diagram or a screenshot will return a plausible-looking label with a
high score. **The model has no way to say "I don't know."**

If your inputs may fall outside everyday photographic subjects, you need an
out-of-distribution check in front of it. This model does not provide one.

### Confidence is not probability

A score of 0.9 does *not* mean "right 90% of the time". Modern deep networks
are systematically overconfident. Our own calibration check, run against a
distribution the model was not trained on, measured an **Expected Calibration
Error of 0.22** — badly calibrated.

Practical consequence: do not build a business rule on a raw confidence
threshold without calibrating first (temperature scaling on a held-out set is
the standard remedy). Use the *ranking* of predictions, which is reliable,
rather than the absolute values, which are not.

### Demographic and geographic bias

ImageNet is drawn largely from English-language web sources and is
well-documented as over-representing North American and European contexts.
Published analyses find accuracy drops markedly on images of household objects
from lower-income countries. Categories relating to people are especially
problematic — parts of the ImageNet "person" subtree were withdrawn by its
maintainers over offensive and non-consensual content.

**Do not use this model to classify people.** It was not built for it, was not
evaluated for it, and the underlying data is not suitable for it.

### One subject at a time

The model assumes a single dominant object. In a cluttered scene it returns
whichever object happens to dominate the centre crop, with no indication that
others were present. Use `/api/v1/detect` for multi-object images.

### Centre-crop blindness

Preprocessing crops the central 87.5%. A subject at the very edge of a wide
photograph can be cropped away entirely before the model ever sees it. The
response includes a `warnings` entry when an image was resized, but it cannot
tell you what was lost.

### Sensitivity to image quality

Accuracy degrades on heavy JPEG compression, motion blur and unusual lighting
— all far more common in real uploads than in a curated benchmark set.

---

## 7. Ethical and operational considerations

* **No image is stored.** The system records a SHA-256 hash of the bytes,
  never the image itself (`db/models.py`).
* **Predictions are logged** with the model version and confidence, for drift
  detection and audit. That log is the only record of what was inferred.
* **Not suitable for consequential decisions about people** — see the bias
  section. There is no human-review workflow built in; if your use case needs
  one, it must live in the calling application.
* **Monitor for drift.** `models/validation/drift.py` compares recent
  prediction and confidence distributions against a baseline. Falling
  confidence with a stable prediction mix usually means the input
  distribution has moved.

---

## 8. Maintenance

| | |
| --- | --- |
| **Registered** | 2026-09-22 |
| **Retrain trigger** | Drift severity `high`, or measured accuracy falling more than 2 points |
| **Regression baseline** | `benchmarks/baselines.json` |
| **Rollback** | Pin the previous version: `{"model_version": "..."}` per request, or re-flag the default in `models/registry.json` and call `POST /api/v1/models/reload` |

### Reproduce

```bash
python scripts/prepare_models.py --only classification
python -m models.validation.validate --model resnet50:1.0.0
python -m models.optimization.benchmark --onnx models/artifacts/resnet50.onnx
```
