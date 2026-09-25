# Model Card: ResNet-50 fine-tuned on Tiny-ImageNet

> A model card says what a model does, how well it does it, and where it
> fails. It exists so whoever deploys it knows what they are deploying, and
> whoever reads its output knows how far to trust it.

| | |
| --- | --- |
| Registry name | `resnet50-tiny-imagenet` |
| Version | `1.0.0` |
| Task | Image classification |
| Status | Registered, not the default classifier (see §6) |
| Endpoint | `POST /api/v1/classify` with `"model_name": "resnet50-tiny-imagenet"` |
| Licence | Base weights from torchvision (BSD-3-Clause); Tiny-ImageNet terms apply to the fine-tuning data |

---

## 1. What it does

Takes one image and returns a ranked list of the most likely categories from
the 200-class Tiny-ImageNet label set, which covers animals, everyday objects,
vehicles and food.

This is the model the brief's fine-tuning requirement produced. It
demonstrates mixed precision, gradient clipping and learning-rate scheduling
on a real dataset, and it is a working classifier. It is not the default for
`/api/v1/classify`; §6 explains why.

---

## 2. How it was trained

```mermaid
flowchart LR
    A["Tiny-ImageNet<br/>200 classes<br/>100,000 images"] --> B["Augmentation<br/>RandAugment · MixUp<br/>CutMix · RandomErasing"]
    B --> C["ResNet-50<br/>ImageNet-1k weights<br/>original stem"]
    C --> D["AdamW 3e-4<br/>cosine + 5% warmup<br/>fp16 AMP · clip 1.0<br/>EMA 0.9998"]
    D --> E["60 epochs<br/>~96 s/epoch<br/>A100-SXM4-40GB"]
    E --> F["78.91% top-1<br/>92.12% top-5"]
```

| Setting | Value |
| --- | --- |
| Base weights | torchvision `IMAGENET1K_V2` |
| Input resolution | 224 × 224 (upsampled from native 64 × 64) |
| Stem | Original ImageNet stem, not adapted |
| Optimiser | AdamW, lr 3e-4, weight decay 5e-2 |
| Schedule | Cosine, 5% linear warmup, 60 epochs |
| Mixed precision | `torch.autocast` fp16 + `GradScaler` |
| Gradient clipping | `clip_grad_norm_`, max norm 1.0, after unscaling |
| Label smoothing | 0.1 |
| Augmentation | RandAugment (2 ops @ 0.4), MixUp α=0.2, CutMix α=1.0, RandomErasing p=0.25 |
| Weight averaging | EMA, decay 0.9998, warmed in |
| Early stopping | Off (`--patience 0`); the run completed all 60 epochs |

All 200 classes and all 100,000 training images were used.
`verify_full_dataset()` refuses to start on a partial dataset.

### Why 224 × 224 for a 64 × 64 dataset

This looks wrong, so here is the explanation.

ResNet-50's stem is a stride-2 7x7 convolution followed by a stride-2 maxpool,
which reduces its input 4x before the first residual block. Feed it 64px and
`layer1` sees a 16x16 map with far too little spatial detail. The usual fix is
to replace the stem with a stride-1 3x3 and drop the maxpool, but that throws
away pretrained stem weights and leaves every later layer running at four
times the spatial area.

A larger input through the original stem uses the network as pretrained. All
three configurations were measured instead of guessed:

| Configuration | s/epoch | Top-1 |
| --- | ---: | ---: |
| 64px, adapted stem, 30 epochs, lr 1e-3 | 82 | 73.98% |
| 128px, original stem, 60 epochs, lr 3e-4 | 38 | 77.66% |
| **224px, original stem, 60 epochs, lr 3e-4, EMA** | **96** | **78.91%** |

The native resolution is the worst of the three and the second slowest. Going
from 128 to 224 buys 1.25 points for 2.5x the compute, which is a small
return: upsampling adds no information, so the ceiling is the dataset rather
than the input size.

---

## 3. Measured performance

### Accuracy

| Metric | Value |
| --- | ---: |
| Top-1 (full 10,000-image val set) | **78.91%** |
| Top-5 | 92.12% |
| Top-1 (2,000-sample validation run) | 78.60% |
| Top-5 (same) | 91.95% |
| Expected Calibration Error | 0.1244 |
| Random baseline | 0.5% |

The gap between the two top-1 figures is the sample subset against the full
validation set, which is ordinary sampling variance.

### The training curve

Plotted by `scripts/plot_training_curves.py` from
`resnet50_training_history.json`, the file the training loop writes:

![Loss, validation accuracy, the learning-rate schedule, and raw versus EMA weights across 60 epochs](../../docs/images/training-curves.png)

The long schedule bought a little, and most of it came at the end:

| Epoch | Top-1 | Gain from here to best |
| ---: | ---: | ---: |
| 1 | 56.52% | +22.39 |
| 2 | 75.12% | +3.79 |
| 10 | 76.97% | +1.94 |
| 20 | 76.56% | +2.35 |
| 30 | 77.03% | +1.88 |
| 40 | 77.90% | +1.01 |
| 50 | 78.42% | +0.49 |
| **57 (best)** | **78.91%** | — |
| 60 | 78.67% | −0.24 |

Transfer learning from ImageNet-1k reaches 75.12% by epoch 2. The remaining 58
epochs add 3.79 points, and roughly half of that arrives after epoch 40 when
the cosine anneal takes the learning rate down two orders of magnitude. The
long schedule is doing real work, which was not obvious in advance: a shorter
run would have stopped around 77%.

Validation loss is nearly flat from epoch 10 (1.688) to epoch 60 (1.659) while
training loss falls from 2.13 to 1.39. The model is fitting the training
distribution considerably faster than it is generalising.

Weight averaging earned its place. The EMA weights scored higher than the live
weights in 53 of 60 epochs, and the best checkpoint is an EMA one. Early in
the run the gap was around two points; by the final anneal, with the learning
rate tiny and the live weights already settled, it narrowed to a few tenths.
The rightmost panel of the chart marks the seven epochs where the raw weights
won.

Non-finite gradients occurred in 8 of the 60 epochs (12, 18, 25, 30, 40, 43,
49, 59). That is normal fp16 behaviour instead of a fault: `GradScaler`
detects the overflow, skips that optimiser step and halves the loss scale. It
is recorded here because an `inf` in a training log looks alarming and is
easier to dismiss with evidence.

Total training time: 1.6 hours, 96 s per epoch.

### Latency by runtime

TensorRT measured on an A100-SXM4-40GB, 200 iterations after 50 warmup runs.
ONNX Runtime measured on the same box's CPU.

```mermaid
xychart-beta
    title "Throughput by runtime (images/second, higher is better)"
    x-axis ["TRT int8", "TRT fp16", "TRT fp32", "ONNX Runtime (CPU)"]
    y-axis "img/s" 0 --> 1300
    bar [1068, 1066, 822, 87]
```

| Runtime | Precision | p50 | p95 | Throughput | Size |
| --- | --- | ---: | ---: | ---: | ---: |
| TensorRT | int8 | 0.920 ms | 1.017 ms | 1068 img/s | 24.1 MB |
| TensorRT | fp16 | 0.990 ms | 1.008 ms | 1066 img/s | 46.0 MB |
| TensorRT | fp32 | 1.298 ms | 1.336 ms | 822 img/s | 91.5 MB |
| ONNX Runtime | fp32 | 11.50 ms | 11.64 ms | 86.9 img/s | 91.2 MB |
| ONNX Runtime | INT8 static | 17.09 ms | 17.51 ms | 58.5 img/s | 23.3 MB |

The two ONNX Runtime rows are CPU numbers, not GPU, **and they are the GPU
host's CPU**, not the development laptop's, and they come from
`BENCHMARKS_GPU.md`. That host is considerably faster: the same model measures
34.3 ms on the laptop. Do not compare these rows with the CPU tables elsewhere
in the repository, which are all laptop numbers. The run requested CUDA but
`onnxruntime-gpu` had no usable CUDAExecutionProvider, and ONNX Runtime falls
back to CPU without raising. The tell is INT8 being slower than fp32, which is
the CPU signature; on a GPU INT8 is faster. Only the TensorRT rows are GPU
figures. The benchmark script detects this and names the report
`BENCHMARKS_GPU_CPU_FALLBACK.md` instead of publishing CPU timings under a
GPU filename.

### Size

```mermaid
xychart-beta
    title "Artifact size (MB, lower is better)"
    x-axis ["ONNX fp32", "TRT fp32 engine", "TRT fp16 engine", "ONNX INT8"]
    y-axis "MB" 0 --> 100
    bar [91.2, 91.5, 46.0, 23.3]
```

---

## 4. Validation results

From `python -m models.validation.validate` against the exported ONNX, loaded
through the same `ModelService` the API uses. Nine of nine checks pass:

| Check | Result |
| --- | --- |
| Artefact integrity | Pass, all artefacts present |
| Determinism | Pass, max diff 0.00e+00 across 3 runs |
| Batch invariance | Pass, max diff 0.00e+00 |
| Output sanity | Pass, no NaN or infinite values |
| Robustness | Pass, 0.0% of predictions flip under σ=0.01 noise |
| Inference errors | Pass, 0 of 2,000 samples failed |
| Accuracy | Pass, top-1 78.60%, top-5 91.95% on 2,000 samples |
| Calibration | Pass, ECE 0.1244 against a 0.15 threshold |
| Latency | Pass, p95 84.4 ms on CPU against a 1,000 ms budget |

ONNX export fidelity against PyTorch: max abs diff 3.81e-06, mean 3.88e-07,
identical top-1 prediction, dynamic batching verified at batch 4.

All three TensorRT engines verify against the fp32 ONNX graph. The check bounds
the difference as a fraction of the reference's peak logit, not as an absolute
number, because logit scale varies by model. Measured against limits of 0.1%,
1% and 10%: fp32 0.05%, fp16 0.40%, int8 1.96%.

The INT8 engine is built from a separate graph, `resnet50-tiny-imagenet_int8_trt
.onnx`, which TensorRT accepts and the CPU one does not. It keeps biases in fp32,
quantizes symmetrically, calibrates by percentile, and leaves the 3-channel stem
convolution unquantized because TensorRT has no INT8 kernel for it. Details are
in docs/TECHNICAL.md.

### INT8 is smaller, slower, and less accurate

Static QDQ quantisation calibrated on 200 real validation images, then
measured on 500 held-out images:

| | fp32 | INT8 static |
| --- | ---: | ---: |
| Size | 91.2 MB | 23.3 MB (3.91x smaller) |
| CPU p50, batch 1 | 34.3 ms | 46.9 ms |
| Top-1 | 80.0% | 63.0% |
| Agreement with fp32 | — | 67.0% |

Latency on the development laptop, from `benchmark_results.json`. Accuracy on
500 held-out validation images put through the API's own preprocessing, from
`quantization_accuracy.json`, so it describes the model as it is served.

Nearly four times smaller, but it disagrees with the full-precision model on a
third of images and costs 17 points of top-1. It is registered and selectable
per request. It is not the default, and these numbers are why.

---

## 5. Preprocessing

This has to match exactly.

| Step | Value |
| --- | --- |
| Resize | Direct resize to 224 × 224 (stretch, no crop) |
| Colour | RGB, alpha composited onto white |
| Scale | 0-255 to 0-1 |
| Normalise | Tiny-ImageNet mean/std (`TINY_IMAGENET_MEAN`, `TINY_IMAGENET_STD`) |

A direct resize, not a centre crop, matching the training `EvalTransform`.
A centre crop at `crop_pct=0.875` against a
training pipeline that resizes directly measures 4.28 apart in normalised
units on identical input, with no error and no crash, just worse
accuracy in production than in validation.

`tests/unit/test_preprocessing_parity.py` (17 tests) reads its expected size
from `TINY_IMAGENET_PREPROCESS` instead of hard-coding a number, so changing
the training resolution cannot desync the two.

---

## 6. Limitations and failure modes

### It only knows 200 things, and they are thumbnails

The label set is 200 Tiny-ImageNet classes. Anything outside them is forced
into the nearest one with a confident-looking score. There is no "I don't
know" output.

Less obviously, it was fine-tuned on 64x64 source images upsampled to 224px.
Its idea of a "dog" is built from thumbnails. Shown a high-resolution
photograph, it sees a rescaled version that is sharper and differently
distributed from anything it trained on.

### Why it is not the default classifier

The default for `/api/v1/classify` is ImageNet-1k ResNet-50: 1,000 classes at
native 224px. For a general-purpose classification API that is the better
model by a wide margin. The brief asks for both a fine-tuning demonstration
and a production API; this model satisfies the first and would do a poor job
of the second.

Making it the default is a one-line registry change if the 200-class label
space is what you want.

### Confidence is not a probability

ECE 0.1244 is inside the 0.15 threshold the validation pipeline enforces, but
it is not good. Mean confidence is 0.662 against 0.786 accuracy, so the model
is noticeably under-confident on this distribution: a score of 0.85 does not
mean 85%. Do not threshold on raw confidence without measuring on your own
data. Temperature scaling would fix this and has not been applied.

### Untested distribution shifts

Not evaluated on medical or satellite imagery, artwork or line drawings, heavy
motion blur, or adversarial inputs. Robustness was tested only against σ=0.01
Gaussian noise, which is a smoke test instead of an adversarial guarantee.

---

## 7. Reproducing it

```bash
python -m models.training.train_classifier \
    --arch resnet50 --epochs 60 --image-size 224 --no-stem-adapt \
    --batch-size 256 --lr 3e-4 --scheduler cosine --warmup-ratio 0.05 \
    --grad-clip 1.0 --label-smoothing 0.1 --patience 0 --ema \
    --device cuda
```

About 1.6 hours on an A100. CPU training is impractical: measured at 3.8 img/s
for the adapted-stem configuration, which is 9.4 days for 30 epochs.

Two settings are traps.

**Early stopping.** Leave it at `--patience 0`. A cosine schedule does most of
its work in the final anneal, so validation accuracy plateaus mid-run as a
matter of course. A patience of 15 ended a 60-epoch run at epoch 32 with the
learning rate still at 86% of maximum, and a patience of 8 ended two earlier
runs around epoch 9. Early stopping suits `--scheduler plateau` or `step`;
with a fixed-length schedule it discards the part that pays.

**Learning rate.** 3e-4, not the 1e-3 that suited the adapted-stem
configuration. With the whole pretrained network intact, 1e-3 makes validation
accuracy regress from 70.5% to 64.8% while training loss keeps falling, with
non-finite gradients appearing. The learning rate has to move when the stem
does.

Long runs on a hosted GPU need a mirror. Pass
`--mirror-dir /content/drive/MyDrive/<folder>/checkpoints` so checkpoints land
somewhere that outlives the container; without it a recycled runtime costs the
whole run.

---

## 8. Maintenance

| | |
| --- | --- |
| Trained | September 2026, A100-SXM4-40GB, TensorRT 11.3 |
| Retrain when | The label space changes, or drift detection flags a sustained shift |
| Drift monitoring | `models/validation/drift.py`: KS test, chi-square, PSI |
| Regression gate | `models/validation/regression.py`, fails the build if accuracy drops against the recorded baseline |
| Owner | See repository maintainers |

Related: [`resnet50-classification.md`](resnet50-classification.md) (the served
default), [`docs/ASSUMPTIONS.md`](../../docs/ASSUMPTIONS.md) §1.2 and §2.1,
[`docs/TECHNICAL.md`](../../docs/TECHNICAL.md) §3.
