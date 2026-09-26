"""COCO mAP for the detector, through the same code path the API serves.

The model card used to quote Ultralytics' published 37.3 mAP. This measures
it: every image goes through ``InferenceService.detect`` (letterbox,
ONNX Runtime, NMS, box rescaling), and the boxes are scored with
pycocotools, the reference implementation. Each runtime is scored on the same
images, so fp32 and INT8 are directly comparable.

Needs the val2017 images and annotations:

    python scripts/download_datasets.py --dataset coco_sample
    # plus annotations/instances_val2017.json from
    # http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Usage::

    python -m models.validation.coco_eval --images 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT = REPO_ROOT / "data" / "coco_val2017"

# Calibration for INT8 uses the first images in filename order, so the
# evaluation slice starts well past them.
DEFAULT_OFFSET = 1000


async def _detections(
    inference: Any,
    images: list[dict[str, Any]],
    image_dir: Path,
    runtime: Any,
    category_ids: dict[str, int],
) -> tuple[list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for info in images:
        response = await inference.detect(
            (image_dir / info["file_name"]).read_bytes(),
            # COCO scoring wants the low-confidence tail: mAP integrates over
            # the whole precision-recall curve.
            confidence_threshold=0.001,
            iou_threshold=0.7,
            max_detections=300,
            model_name="yolov8n",
            runtime=runtime,
            use_cache=False,
        )
        if response.model.runtime != runtime:
            raise RuntimeError(f"asked for {runtime.value}, served by {response.model.runtime}")
        for det in response.detections:
            box = det.box
            results.append(
                {
                    "image_id": info["id"],
                    "category_id": category_ids[det.label],
                    "bbox": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
                    "score": det.confidence,
                }
            )
    return results, (time.perf_counter() - started) * 1000 / max(len(images), 1)


def _score(coco: Any, detections: list[dict[str, Any]], image_ids: list[int]) -> dict[str, float]:
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"map_50_95": 0.0, "map_50": 0.0, "map_75": 0.0}
    evaluator = COCOeval(coco, coco.loadRes(detections), "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {"map_50_95": float(stats[0]), "map_50": float(stats[1]), "map_75": float(stats[2])}


def evaluate(root: Path, num_images: int, offset: int) -> dict[str, Any]:
    from pycocotools.coco import COCO

    from api.config import Settings
    from api.models.schemas import RuntimeFormat
    from api.services.cache_service import CacheService
    from api.services.inference_service import InferenceService
    from api.services.model_service import ModelService

    coco = COCO(str(root / "annotations" / "instances_val2017.json"))
    image_dir = root / "val2017"
    images = sorted(coco.loadImgs(coco.getImgIds()), key=lambda i: i["file_name"])
    images = [i for i in images if (image_dir / i["file_name"]).exists()][
        offset : offset + num_images
    ]
    image_ids = [i["id"] for i in images]
    category_ids = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    cfg = Settings(environment="test", cache_enabled=False)
    models = ModelService(cfg)
    inference = InferenceService(models, CacheService(cfg), cfg)

    runs: dict[str, Any] = {}
    for runtime in (RuntimeFormat.ONNX, RuntimeFormat.ONNX_INT8):
        detections, per_image_ms = asyncio.run(
            _detections(inference, images, image_dir, runtime, category_ids)
        )
        runs[runtime.value] = {
            **_score(coco, detections, image_ids),
            "detections": len(detections),
            "mean_ms_per_image": round(per_image_ms, 1),
        }
    models.unload_all()

    return {
        "model": "yolov8n:1.0.0",
        "dataset": "COCO val2017",
        "images": len(images),
        "image_offset": offset,
        "note": (
            "Images in filename order, starting past the INT8 calibration set. "
            "Scored with pycocotools; confidence threshold 0.001, NMS IoU 0.7, up to "
            "300 detections, as COCO evaluation expects. Timing includes preprocessing "
            "and NMS."
        ),
        "runtimes": runs,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "coco_eval.json"
    )
    args = parser.parse_args()

    if not (args.root / "annotations" / "instances_val2017.json").is_file():
        print(f"error: no COCO annotations under {args.root}; see the module docstring")
        return 2

    report = evaluate(args.root, args.images, args.offset)
    for name, run in report["runtimes"].items():
        print(
            f"{name:<10} mAP50-95 {run['map_50_95']:.3f}  mAP50 {run['map_50']:.3f}  "
            f"{run['mean_ms_per_image']} ms/image"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
