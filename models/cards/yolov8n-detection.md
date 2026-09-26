# Model Card: YOLOv8n (object detection)

| | |
| --- | --- |
| Registry name | `yolov8n`, version `1.0.0`, default for detection |
| Endpoint | `POST /api/v1/detect` |
| Weights | Ultralytics COCO checkpoint, no fine-tuning |
| Licence | **AGPL-3.0**, see section 6 |

## 1. What it does

Returns every object it finds among the 80 COCO categories, each with a class,
a confidence and a box in the pixel coordinates of the uploaded image. People,
vehicles, animals, common household and street objects.

## 2. Why this model

YOLOv8n predicts all boxes in one forward pass, which is what makes CPU
detection feasible at all. It is anchor-free, which keeps the
postprocessing we implement ourselves (NMS and un-letterboxing, in
`api/services/inference_service.py`) simple, and it exports to ONNX in one
call. DETR-style detectors need custom operators or tracing workarounds.

Nano is the smallest variant: 3.2 M parameters, 12.1 MB. It is still the
slowest model in this service. YOLOv8m is 9x the compute (78.9 against 8.7
GFLOPs) and, by Ultralytics' published CPU timings, about 3x the latency, for
50.2 mAP50-95 against 37.3. The price of nano is accuracy on small and distant
objects (section 5). `--detector yolov8s` in
`scripts/prepare_models.py` swaps it.

## 3. Preprocessing

| Step | Value |
| --- | --- |
| Resize | Letterbox to 640x640, aspect ratio kept, grey (114) padding |
| Scale | 0-255 to 0-1 |
| Normalise | None; YOLO normalises internally |

Letterboxing matters: squashing a wide photo into a square distorts every box.
`scale_boxes_to_original()` maps boxes back and has a round-trip test.
ImageNet mean/std, the reflex from the classification path, wrecks the output.
That is why each model's preprocessing lives in the registry beside it.

## 4. Measured performance

### Accuracy on COCO

Measured with pycocotools on 500 val2017 images (filename order, starting at
image 1000, clear of the INT8 calibration set), through the same
`InferenceService.detect` the API uses. `benchmarks/reports/coco_eval.json`.

| Runtime | mAP50-95 | mAP50 | mAP75 |
| --- | ---: | ---: | ---: |
| ONNX fp32 | 0.392 | 0.536 | 0.430 |
| ONNX INT8 | 0.388 | 0.540 | 0.429 |

Ultralytics publishes 37.3 mAP50-95 on the full 5,000-image set. 39.2 on this
500-image slice is in line with it.

### Latency

Intel Core Ultra 7 155H, CPU only, ONNX Runtime 1.20.1, 100 runs per case
interleaved with the other models (`benchmarks/reports/BENCHMARKS.md`).

| Runtime | Batch | p50 | p95 | p99 | Throughput | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX fp32 | 1 | 113.1 ms | 226.2 ms | 307.5 ms | 7.3 img/s | 12.1 MB |
| ONNX fp32 | 4 | 396.9 ms | 670.3 ms | 829.3 ms | 9.5 img/s | 12.1 MB |
| ONNX INT8 | 1 | 150.9 ms | 256.2 ms | 306.9 ms | 6.2 img/s | 3.3 MB |
| ONNX INT8 | 4 | 523.5 ms | 975.2 ms | 1181.8 ms | 7.2 img/s | 3.3 MB |

Single images are well inside the one-second budget. INT8 at batch 4 is not,
at p99, which is one reason batches go through the async endpoint.

### INT8

INT8 is 3.67x smaller and 0.4 mAP points below fp32 (0.388 against 0.392),
but 1.33x slower on this CPU. The likely reason, not separately profiled, is
that only the convolutions are quantized, so the fp32 decode head and the
extra quantize and dequantize steps outweigh the gain. fp32 is the default; INT8 is available per request for memory-bound
deployments.

The first INT8 model detected nothing. YOLOv8's head concatenates box
coordinates (0 to 640) and class scores (0 to 1) into one tensor, and
quantizing that Concat gave both one int8 scale, about 2.5 per step, so every
class score rounded to zero. The quantization report still said "100%
agreement", because agreement was only computed for classifier outputs. The
current model quantizes only the convolutions, uses uint8 (portable across
x86 CPUs, with and without VNNI), and is calibrated on 100 COCO images with
percentile clipping and the detector's own letterbox preprocessing. On 500
held-out COCO images its most confident class matches fp32 on 91.8%
(`benchmarks/reports/int8_fidelity.json`). CI runs every INT8 model against its
fp32 twin on real photos (`tests/integration/test_quantized_fidelity.py`).

### Validation

`python -m models.validation.validate --model yolov8n:1.0.0`, recorded in
`benchmarks/reports/validation.json`: artifacts present, deterministic across
three runs, batch-invariant, no NaN or Inf, p95 latency 167.6 ms against
a 1,000 ms budget. The batch-invariance check once caught an export with a
fixed batch dimension that would have failed every detection batch job; the
export now uses `dynamic=True`.

## 5. Limitations

- **80 classes and nothing else.** There is no "unknown" output. Industrial
  equipment, medical images or documents get nothing or confident nonsense.
- **Small, distant and occluded objects are missed.** This is the main cost of
  nano. Lowering `confidence_threshold` trades misses for false positives; a
  larger model is the fix.
- **640x640 loses detail.** A 4000x3000 photo is shrunk more than 6x, so
  a 40-pixel object becomes 6 pixels. Tiled inference would fix it and is not
  implemented.
- **NMS can merge real neighbours.** Two people standing close can overlap past
  `iou_threshold`, and one is dropped. No single threshold suits every scene.
- **Confidence is not calibrated.** Use it to rank, not as a probability.
- **COCO's biases carry over.** It is web-sourced and skews Western and urban.
- **Not for surveillance, identification or safety-critical use.** It says a
  person is present; it cannot say who, and its miss rate rules out uses where
  a miss causes harm. No image is stored, only a hash.

## 6. Licence

Ultralytics YOLOv8 is AGPL-3.0. Running it as a network service obliges you to
offer your application's source to its users. For commercial use, buy an
Ultralytics Enterprise licence, swap in a permissively licensed detector (a
registry change, since the serving code is architecture-agnostic), or publish
your source.

## 7. Reproduce

```bash
python scripts/prepare_models.py --only detection
python -m models.optimization.quantize --onnx models/artifacts/yolov8n.onnx --mode static \
    --calibration-dir data/coco_val2017/val2017 --preset yolo_640 --op-types Conv --num-calibration 100
python -m models.validation.coco_eval --images 500
python -m models.validation.validate --model yolov8n:1.0.0
```
