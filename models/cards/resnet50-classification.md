# Model Card: ResNet-50 (image classification)

| | |
| --- | --- |
| Registry name | `resnet50`, version `1.0.0`, default for classification |
| Endpoint | `POST /api/v1/classify` |
| Weights | torchvision `IMAGENET1K_V2` (BSD-3-Clause); ImageNet terms apply to the data |

## 1. What it does

Returns the most likely of the 1,000 ImageNet categories for one photograph,
with a confidence for each: everyday objects, animals, vehicles, food. It says
what the main subject is, not where it is or how many there are; that is
`POST /api/v1/detect`.

The fine-tuned Tiny-ImageNet model the brief asks for is a separate registry
entry with its own [card](resnet50-tiny-imagenet.md). It is not the default,
because 200 classes learned from 64x64 thumbnails make a poor general-purpose
classifier ([ASSUMPTIONS.md](../../docs/ASSUMPTIONS.md) §1.4).

## 2. Why this model

ResNet-50 has 25.6 M parameters. A Vision Transformer of similar accuracy needs
about three to four times the compute, and this service is CPU-first. Every
ResNet operation has a well-supported ONNX equivalent and it quantizes without
special handling, which matters when the pipeline has to be reproducible.
ConvNeXt or EfficientNetV2 would be two to four points more accurate; that was
traded for latency headroom and a clean export. The serving code does not care
which backbone it gets, so swapping is a new registry entry.

## 3. Preprocessing

| Step | Value |
| --- | --- |
| Resize | Shortest side to 256 |
| Crop | Centre 224x224 |
| Colour | RGB, alpha composited onto white |
| Normalise | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |

These are the statistics the weights were trained with. Different ones cost
accuracy with no error. They live in `CLASSIFICATION_PREPROCESS` and travel
with the model through the registry.

## 4. Measured performance

### Accuracy

torchvision reports 80.86% top-1 and 95.43% top-5 on the ImageNet-1k
validation set for these weights. It is not re-measured here: that set needs a
registered account. The validation pipeline refuses to score this model on
Tiny-ImageNet, since 1,000 and 200 class indices mean different things and the
number would be meaningless.

The ONNX export matches PyTorch to a maximum absolute difference of 2.86e-06
(`benchmarks/reports/onnx_export.json`).

### Latency

Intel Core Ultra 7 155H, CPU only, ONNX Runtime 1.20.1 and PyTorch 2.9.0, 100 runs per case
interleaved with the other models (`benchmarks/reports/BENCHMARKS.md`).

| Runtime | Batch | p50 | p95 | p99 | Throughput | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch fp32 | 1 | 79.9 ms | 230.5 ms | 268.2 ms | 9.3 img/s | 97.5 MB |
| PyTorch fp32 | 4 | 201.3 ms | 573.1 ms | 625.4 ms | 15.0 img/s | 97.5 MB |
| ONNX fp32 | 1 | 30.1 ms | 107.6 ms | 143.4 ms | 21.9 img/s | 97.4 MB |
| ONNX fp32 | 4 | 105.1 ms | 260.1 ms | 307.0 ms | 29.8 img/s | 97.4 MB |
| ONNX INT8 | 1 | 13.9 ms | 65.6 ms | 146.1 ms | 41.0 img/s | 24.9 MB |
| ONNX INT8 | 4 | 43.6 ms | 132.9 ms | 161.1 ms | 65.8 img/s | 24.9 MB |

p99 for a single image is 143.4 ms in ONNX fp32, well inside the one-second budget.

### INT8

INT8 (uint8, percentile calibration) is 3.92x smaller and 2.2x faster than
fp32 at batch 1 on this CPU: 13.9 ms against 30.1 ms p50. Without labels its
accuracy cannot be measured here, but agreement can: on 500 held-out images
INT8 picks the same top class as fp32 86.2% of the time
(`benchmarks/reports/int8_fidelity.json`). Because a seventh of answers change
and the accuracy cost is unmeasured, fp32 stays the default; INT8 is available
with `"runtime": "onnx_int8"`.

### Validation

All checks pass in `benchmarks/reports/validation.json`: artifacts present,
deterministic across three runs, batch-invariant, no NaN or Inf, no prediction
flips under imperceptible noise (sigma 0.01), p95 71.1 ms against a 1,000 ms
budget.

## 5. Limitations

- **It only knows 1,000 things.** A microscope slide, an X-ray or a screenshot
  gets a plausible label with a high score. It cannot say "I don't know"; put
  an out-of-distribution check in front of it if your inputs can stray.
- **Confidence is not probability.** Deep networks are overconfident. Use the
  ranking; calibrate before building a rule on a threshold.
- **Bias.** ImageNet over-represents North American and European contexts,
  and accuracy drops on household objects from lower-income countries. Parts of
  its "person" subtree were withdrawn by its maintainers. Do not use this model
  to classify people.
- **One subject at a time.** In a cluttered scene it names whatever dominates
  the centre crop.
- **Centre crop.** A subject at the edge of a wide photo can be cropped away
  before the model sees it. The response warns that the image was resized, not
  what was lost.
- **Image quality.** Heavy compression, blur and odd lighting all hurt, and are
  more common in real uploads than in benchmark sets.

No image is stored, only a SHA-256 hash. Predictions are logged with model
version and confidence, which is what drift detection reads.

## 6. Maintenance

| | |
| --- | --- |
| Regression baseline | `benchmarks/baselines.json` |
| Rollback | `{"model_version": "..."}` per request, or re-flag the default and `POST /api/v1/models/reload` |

```bash
python scripts/prepare_models.py --only classification
python -m models.validation.validate --model resnet50:1.0.0
```
