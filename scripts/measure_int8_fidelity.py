"""How closely each INT8 model tracks its fp32 twin, on real images.

Accuracy needs labels, and only the fine-tuned classifier has a labelled set
here (the A/B test and COCO evaluation cover accuracy where labels exist).
Agreement needs no labels, so it can be measured for every model:

* classifiers: top-1 agreement
* embedding model: cosine similarity between the two vectors
* detector: the most confident detected class (or "nothing") agrees

Images go through each model's registered preprocessing, as the API serves
them. Classifiers use Tiny-ImageNet validation images from index 1000 on,
past the calibration images; the detector uses COCO val2017 from index 1000.

    python scripts/measure_int8_fidelity.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.models.schemas import RuntimeFormat
from api.services.model_service import ModelService
from api.utils.image_processing import preprocess
from models.optimization.quantize import _dominant_class

IMAGES = 500
OFFSET = 1000
TINY = REPO_ROOT / "data" / "tiny-imagenet-200" / "tiny-imagenet-200" / "val" / "images"
COCO = REPO_ROOT / "data" / "coco_val2017" / "val2017"


def _files(directory: Path) -> list[Path]:
    files = sorted(f for f in directory.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})
    return files[OFFSET : OFFSET + IMAGES]


def main() -> int:
    service = ModelService()
    report: dict[str, object] = {
        "images_per_model": IMAGES,
        "offset": OFFSET,
        "generated_at": datetime.now(UTC).isoformat(),
        "models": {},
    }
    for entry in service.list_entries():
        if RuntimeFormat.ONNX_INT8.value not in entry.artifacts:
            continue
        fp32 = service.load(entry, RuntimeFormat.ONNX)
        int8 = service.load(entry, RuntimeFormat.ONNX_INT8)
        assert int8.runtime.format == RuntimeFormat.ONNX_INT8, f"{entry.key} INT8 fell back"
        files = _files(COCO if entry.task.value == "detection" else TINY)

        scores: list[float] = []
        for path in files:
            x = preprocess(path.read_bytes(), fp32.preprocess_config).array
            a = fp32.runtime.infer(x)[0]
            b = int8.runtime.infer(x)[0]
            if entry.task.value == "detection":
                scores.append(float(_dominant_class(a) == _dominant_class(b)))
            elif entry.task.value == "similarity":
                va, vb = a.reshape(-1), b.reshape(-1)
                scores.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))))
            else:
                scores.append(float(a[0].argmax() == b[0].argmax()))

        metric = {
            "detection": "dominant_class_agreement",
            "similarity": "mean_cosine",
            "classification": "top1_agreement",
        }[entry.task.value]
        result = {
            "metric": metric,
            "value": round(float(np.mean(scores)), 4),
            "images": len(scores),
            "dataset": "COCO val2017" if entry.task.value == "detection" else "Tiny-ImageNet val",
        }
        if metric == "mean_cosine":
            result["min_cosine"] = round(float(np.min(scores)), 4)
        report["models"][entry.key] = result  # type: ignore[index]
        print(f"{entry.key:<32} {metric:<26} {result['value']}")

    out = REPO_ROOT / "benchmarks" / "reports" / "int8_fidelity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
