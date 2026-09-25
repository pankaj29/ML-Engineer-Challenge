# Model Card, YOLOv8n (Object Detection)

| | |
| --- | --- |
| **Registry name** | `yolov8n` |
| **Version** | `1.0.0` |
| **Task** | Object detection |
| **Status** | Active, default for the detection task |
| **Serving endpoint** | `POST /api/v1/detect` |
| **Licence** | **AGPL-3.0** (Ultralytics), see section 7, this has commercial implications |

---

## 1. What it does

Given a photograph, it returns every recognised object it can find, each with
a class name, a confidence score and a bounding box in the pixel coordinates
of the **original uploaded image**.

It recognises the **80 COCO categories**: people, vehicles, animals, common
household and street objects.

Where classification answers "what is this a picture of?", detection answers
"what things are in it, and where?", including several objects at once, and
several instances of the same class.

---

## 2. Architecture and why it was chosen

**YOLOv8n** ("nano"): ~3.2 M parameters, 12.1 MB as ONNX.

YOLO, "You Only Look Once", predicts every box in a single forward pass. The
older two-stage approach proposes regions first and classifies each one after.
One pass over the image is why it is fast enough to
run on a CPU at all.

YOLOv8 in particular is *anchor-free*: it predicts box coordinates directly
without adjusting a set of predefined box shapes. That removes a whole
category of tuning (choosing anchor sizes for your dataset) and simplifies the
postprocessing, which matters because we implement that postprocessing
ourselves in `api/services/inference_service.py`.

Why the nano variant:

* **Detection is expensive.** Even nano is the slowest model here at 97.1 ms
  p50. YOLOv8m would be roughly 4x that, pushing a batch of
  four past the latency budget on CPU.
* **It exports cleanly to ONNX.** This is not a given. DETR-family detectors
  and several two-stage architectures need custom operators or tracing
  workarounds; YOLOv8 exports with one call.
* **Size.** 12 MB means the model can be shipped, cached and loaded quickly.

The trade-off is a real one. Nano is the least accurate
YOLOv8 variant: roughly **37.3 mAP50-95** on COCO, against ~50.2 for the
medium variant. In practice that means it misses small, distant and
partially-hidden objects that a larger model would find. Section 6 is explicit
about this. Moving to `yolov8s` or `yolov8m` is a single argument change
(`--detector yolov8m`) when a GPU is available.

---

## 3. Training data

**COCO 2017** ("Common Objects in Context"): ~118,000 training images with
~860,000 labelled object instances across 80 categories. Photographs of
everyday scenes, cluttered and unposed, which is why COCO models
generalise to real photographs better than models trained on isolated objects.

The weights are Ultralytics' published COCO checkpoint. **No fine-tuning was
performed for this project**; the brief asks for a detector on a COCO subset,
and the pretrained COCO weights are exactly that, evaluated on the full set
and not a subset.

### Preprocessing (must match exactly)

| Step | Value |
| --- | --- |
| Resize | **Letterbox** to 640 x 640, aspect ratio preserved, remainder padded |
| Padding | Grey, value 114 |
| Colour | RGB |
| Scale | 0-255 → 0-1 |
| Normalise | **None**, YOLO normalises internally |

Letterboxing is not optional. Squashing a 1920x1080 photo into a square
compresses every object horizontally, and the predicted boxes are then wrong
in a way that is hard to spot and impossible to undo. The inverse transform
that maps boxes back to original coordinates lives in
`scale_boxes_to_original()` and is covered by a round-trip test.

Note also that YOLO expects **un-normalised** 0-1 input. Applying ImageNet
mean/std here, the reflex from the classification path, silently wrecks the
output. This is why preprocessing config travels with each model in the
registry, with no global state.

---

## 4. Measured performance

Intel Core Ultra 7 155H, 22 logical cores, CPU only, ONNX Runtime 1.26.0.

### Latency

| Format | Batch | p50 | p95 | p99 | Throughput | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX fp32 | 1 | **97.1 ms** | 154.4 ms | 166.1 ms | 9.6 img/s | 12.1 MB |
| ONNX fp32 | 4 | 325.8 ms | 444.5 ms | 473.7 ms | 12.2 img/s | 12.1 MB |
| ONNX INT8 (static) | 1 | 223.6 ms | 309.4 ms | 337.6 ms | 4.4 img/s | **3.4 MB** |
| ONNX INT8 (static) | 4 | 887.5 ms | 1133.0 ms | 1140.2 ms | 4.3 img/s | 3.4 MB |

p99 at batch 1 is 166 ms, inside the 1-second budget.
Note that **batch 4 at p99 exceeds 1 second**, batching detection trades
per-image efficiency for worse tail latency, which is why the batch endpoint
is asynchronous.

INT8 is 3.55x smaller and 2.3x slower here; fp32 remains the default.

### Detection accuracy

Published COCO val2017 figures for these weights:

| Metric | Value |
| --- | --- |
| mAP50-95 | 37.3 |
| mAP50 | ~52.6 |

Not independently re-measured. Doing so needs the COCO validation set
(~1 GB) plus the official evaluation protocol. What *was* verified is
end-to-end correctness on real photographs: on the standard test image the
model finds 4 people and 1 bus, with boxes correctly placed in original image
coordinates, and the fp32 and INT8 variants agree exactly.

---

## 5. Validation results

| Check | Result |
| --- | --- |
| Artefact integrity | Pass |
| Determinism | Pass, identical output across 3 runs |
| Batch invariance | **Pass, after a fix.** The first export had a fixed batch dimension and crashed on any batch > 1. The validation pipeline caught it; the export now uses `dynamic=True`. |
| Output sanity | Pass, no NaN/Inf |
| Latency | Pass, p95 116.1 ms |

The batch-invariance failure deserves a mention: it would have made the
`/api/v1/batch` endpoint fail for every detection job, and nothing else in the
suite would have noticed.

---

## 6. Limitations and failure modes

### 80 categories, and nothing else

Anything outside COCO's 80 classes is either missed entirely or labelled as
the nearest class. There is no "unknown object" output. A photograph of
industrial equipment, medical imagery or a document will return either nothing
or confident nonsense.

### Nano misses small and distant objects

This is the single most important practical limitation. The nano variant is
substantially weaker than larger YOLOv8 models on:

* **small objects**, anything occupying a small fraction of the frame;
* **distant objects**, a crowd at the back of a scene;
* **occluded objects**, a person half behind a car.

If recall on small objects matters, use a larger variant. Do not compensate by
lowering `confidence_threshold`: that trades misses for false positives rather
than finding the objects.

### Fixed 640x640 input loses detail

Every image is letterboxed into 640x640. A 4000x3000 photograph is downscaled
by more than 6x, and objects that were 40 pixels across become 6 pixels and
effectively vanish. For high-resolution imagery where small objects matter,
tiled inference (running the detector over overlapping crops) is the standard
answer; it is not implemented here.

### NMS merges truly overlapping objects

Non-maximum suppression removes boxes that overlap a higher-scoring box by
more than `iou_threshold`. In a crowd, two people standing close together can
legitimately overlap more than the threshold, and one is discarded. Raising
`iou_threshold` keeps them, at the cost of duplicate boxes elsewhere. There is
no setting that is correct for all scenes, tune it for yours.

### Confidence is not calibrated

As with the classifier, scores are useful for ranking, not as probabilities.

### Inherits COCO's biases

COCO is web-sourced and skews Western and urban. Object categories reflect
what was common in that data. The `person` class has the same ethical
considerations noted in the classification card.

---

## 7. Licensing, read this before commercial use

Ultralytics YOLOv8 is AGPL-3.0, a strong copyleft licence. If you
run it as a network service, the AGPL requires you to offer the complete
corresponding source of your application to its users.

This matters for a commercial deployment and is a genuine decision, not a
formality. The options are:

1. **Buy an Ultralytics Enterprise licence**, the route they intend for
   closed-source commercial use.
2. **Swap the detector** for a permissively licensed one. The serving code is
   architecture-agnostic; a detector exporting to ONNX with a comparable
   output layout is a registry change.
3. **Comply with the AGPL** and publish your source.

Flagged here at the top, because licence problems are cheap to fix at
the design stage and expensive to fix after launch.

---

## 8. Ethical and operational considerations

* **No image is stored**, only a hash.
* **Not suitable for surveillance or identification.** It detects that a
  person is present; it cannot and must not be used to identify anyone.
* **Not suitable for safety-critical use.** Its miss rate on small and
  occluded objects makes it unfit for anything where a missed detection causes
  harm.

---

## 9. Maintenance

| | |
| --- | --- |
| **Registered** | 2026-09-22 |
| **Upgrade path** | `python scripts/prepare_models.py --only detection --detector yolov8s` |
| **Rollback** | Pin `model_version` per request, or re-flag the default and reload |

### Reproduce

```bash
python scripts/prepare_models.py --only detection
python -m models.validation.validate --model yolov8n:1.0.0
```
