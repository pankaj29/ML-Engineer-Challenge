# Model Card: ResNet-50 fine-tuned on Tiny-ImageNet

| | |
| --- | --- |
| Registry name | `resnet50-tiny-imagenet`, version `1.0.0` |
| Status | Registered, not the default classifier (section 5) |
| Endpoint | `POST /api/v1/classify` with `"model_name": "resnet50-tiny-imagenet"` |
| Weights | Fine-tuned from torchvision `IMAGENET1K_V2` (BSD-3-Clause) |

## 1. What it does

Returns the most likely of Tiny-ImageNet's 200 classes (animals, everyday
objects, vehicles, food) for one image. It is the model the brief's
fine-tuning requirement produced: mixed precision, gradient clipping and
learning-rate scheduling on a real dataset, trained to completion.

## 2. How it was trained

| Setting | Value |
| --- | --- |
| Data | All 200 classes, all 100,000 training images; `verify_full_dataset()` refuses a partial set |
| Input | 224x224, upsampled from the native 64x64, original ImageNet stem |
| Optimiser | AdamW, lr 3e-4, weight decay 0.05 |
| Schedule | Cosine with 5% linear warmup, 60 epochs, stepped per batch |
| Mixed precision | `torch.autocast` fp16 with `GradScaler` |
| Gradient clipping | `clip_grad_norm_` at 1.0, after unscaling |
| Regularisation | Label smoothing 0.1, RandAugment, MixUp, CutMix, RandomErasing (all in `models/training/augmentation.py`) |
| Weight averaging | EMA, decay 0.9998 |
| Hardware | One A100-SXM4-40GB, 95.6 s per epoch, 1.6 hours in total |

Why 224px for a 64px dataset: ResNet-50's stem shrinks its input 4x before the
first residual block, so at 64px `layer1` sees a 16x16 map. The usual fix
replaces the stem, which discards its pretrained weights and makes every later
layer run on four times the area. Upsampling and keeping the pretrained stem
uses the network as it was trained. Shorter runs at 64px (adapted stem) and
128px during development both scored lower; their logs were not kept, so they
are not quoted.

### The training curve

From `benchmarks/reports/resnet50_training_history.json`, plotted by
`scripts/plot_training_curves.py`:

![Loss, validation accuracy, learning rate, and raw versus EMA weights over 60 epochs](../../docs/images/training-curves.png)

| Epoch | Top-1 | Still to gain |
| ---: | ---: | ---: |
| 1 | 56.52% | 22.39 |
| 2 | 75.12% | 3.79 |
| 10 | 76.97% | 1.94 |
| 30 | 77.03% | 1.88 |
| 40 | 77.90% | 1.01 |
| 50 | 78.42% | 0.49 |
| 57 (best) | 78.91% | |
| 60 | 78.67% | |

Transfer learning does most of the work by epoch 2. The next 55 epochs add
3.79 points, and about half of that arrives after epoch 40, when the cosine
anneal takes the learning rate down two orders of magnitude. Validation loss
barely moves after epoch 10 (1.688 to 1.659) while training loss keeps falling
(2.126 to 1.390).

EMA weights were the better of the two in 53 of 60 epochs, and the best
checkpoint is an EMA one. `GradScaler` visibly did its job: the end-of-epoch
loss scale drops at epochs 12, 18, 25, 43 and 49, each time it skipped a step
with non-finite gradients and halved the scale.

## 3. Measured performance

### Accuracy

The exported ONNX model, served through `ModelService` with the API's own
preprocessing, on all 10,000 validation images
(`benchmarks/reports/validation.json`):

| Metric | Value |
| --- | ---: |
| Top-1 | **78.91%** |
| Top-5 | 92.12% |
| Expected calibration error | 0.128 |
| Random baseline | 0.5% |

This is exactly the figure the training loop recorded in PyTorch, so export
and serving preprocessing lose nothing. The export itself matches PyTorch to a
maximum absolute difference of 3.81e-06 (`benchmarks/reports/onnx_export.json`).

### Latency

Intel Core Ultra 7 155H, CPU only, ONNX Runtime 1.20.1, 100 runs per case
interleaved with the other models (`benchmarks/reports/BENCHMARKS.md`):

| Runtime | Batch | p50 | p95 | p99 | Throughput | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX fp32 | 1 | 66.6 ms | 139.0 ms | 176.6 ms | 15.2 img/s | 91.2 MB |
| ONNX fp32 | 4 | 207.5 ms | 311.7 ms | 333.1 ms | 18.7 img/s | 91.2 MB |
| ONNX INT8 | 1 | 76.8 ms | 156.5 ms | 268.9 ms | 12.0 img/s | 23.3 MB |
| ONNX INT8 | 4 | 340.7 ms | 560.2 ms | 687.7 ms | 11.3 img/s | 23.3 MB |

TensorRT on an A100-SXM4-40GB, TensorRT 11.3.0.99, batch 1
(`benchmarks/reports/tensorrt.json`):

| Precision | p50 | p95 | Throughput | Engine | Max abs diff vs ONNX |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 1.298 ms | 1.336 ms | 822 img/s | 91.5 MB | 2.62e-03 |
| fp16 | 0.990 ms | 1.008 ms | 1066 img/s | 46.0 MB | 2.19e-02 |
| INT8 | 0.920 ms | 1.017 ms | 1068 img/s | 24.1 MB | 1.10e-01 |

All three engines passed verification against the fp32 ONNX graph on a real
photograph. INT8 is 1.41x faster than fp32 and a quarter of its size; at batch
1 it is level with fp16, because this model at batch 1 is bound by memory
traffic, not arithmetic.

### INT8 on CPU

A paired A/B test on all 10,000 validation images, fp32 against static INT8
through the serving path (`benchmarks/reports/ab_test.json`):

| | fp32 | INT8 |
| --- | ---: | ---: |
| Top-1 | ⟦AB_FP32⟧ | ⟦AB_INT8⟧ |
| p95 latency | ⟦AB_FP32_P95⟧ | ⟦AB_INT8_P95⟧ |

⟦AB_SENTENCE⟧ INT8 is 3.91x smaller, and on this CPU also slower. It stays
available per request and is not the default.

### Validation

All nine checks pass: artifacts, determinism across three runs,
batch invariance, output sanity, robustness to sigma 0.01 noise, zero
inference errors over 10,000 images, accuracy, calibration, and p95 latency of
112.7 ms against a 1,000 ms budget.

## 4. Preprocessing

| Step | Value |
| --- | --- |
| Resize | Direct resize to 224x224 (no crop) |
| Colour | RGB, alpha composited onto white |
| Normalise | Tiny-ImageNet mean and std |

A direct resize, matching the training `EvalTransform`. A centre crop would
give the model different pixels from the ones it was validated on, silently.
`tests/unit/test_preprocessing_parity.py` reads its expected values from
`TINY_IMAGENET_PREPROCESS`, so training and serving cannot drift apart.

## 5. Limitations

- **It knows 200 things, learned from thumbnails.** Anything else is forced
  into the nearest class. A sharp high-resolution photo is also unlike the
  upsampled 64px images it learned from.
- **Not the default classifier.** ImageNet-1k ResNet-50 has 1,000 classes at
  native resolution and is the better general-purpose model. Making this one
  the default is a one-line registry change.
- **Underconfident.** Mean confidence is 0.661 against 0.789 accuracy on the
  validation set, so a score of 0.85 does not mean 85%. Temperature scaling
  would fix it and is not applied.
- **Untested shifts.** Not evaluated on medical or satellite images, artwork,
  heavy blur or adversarial input. The noise check is a smoke test.

## 6. Reproduce

```bash
python -m models.training.train_classifier \
    --arch resnet50 --epochs 60 --image-size 224 --no-stem-adapt \
    --batch-size 256 --lr 3e-4 --scheduler cosine --warmup-ratio 0.05 \
    --grad-clip 1.0 --label-smoothing 0.1 --patience 0 --ema --device cuda
```

About 1.6 hours on an A100. On the laptop CPU the same model trains at
⟦CPU_R50_IPS⟧ images per second (`benchmarks/reports/cpu_training_throughput.json`),
which is why it was trained on a GPU. For hosted GPUs, pass `--mirror-dir` so
checkpoints outlive the container.

Leave early stopping off (`--patience 0`). A cosine schedule does much of its
work in the final anneal, so validation accuracy plateaus mid-run as a matter
of course, and stopping there throws away the part that pays.
