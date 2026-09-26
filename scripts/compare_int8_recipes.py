"""Compare INT8 recipes for every model, to justify the one that ships.

Four static-quantization recipes, each built from the fp32 model with the
model's own serving preprocessing and 100 calibration images:

* ``s8s8_minmax``: int8 activations and weights, MinMax calibration
* ``s8s8_minmax_reduce_range``: the same with 7-bit weights
* ``u8u8_minmax``: uint8 activations and weights, MinMax calibration
* ``u8u8_percentile``: uint8, percentile calibration (what ships)

Scored on held-out images, never the calibration set:

* resnet50-tiny-imagenet: top-1 accuracy on 2,000 labelled validation images
* resnet50: top-1 agreement with fp32 on 500 Tiny-ImageNet images
* resnet50-embed: mean and minimum cosine to fp32 on the same 500
* yolov8n: agreement on the most confident class, 500 COCO images

    python scripts/compare_int8_recipes.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.services.model_service import PREPROCESS_PRESETS
from api.utils.image_processing import preprocess
from models.optimization.quantize import (
    _dominant_class,
    _OnnxCalibrationReader,
    iter_calibration_images,
)
from models.training.dataset import TinyImageNetTrain, TinyImageNetVal, find_dataset_root

ARTIFACTS = REPO_ROOT / "models" / "artifacts"
TINY = REPO_ROOT / "data" / "tiny-imagenet-200" / "tiny-imagenet-200" / "val" / "images"
COCO = REPO_ROOT / "data" / "coco_val2017" / "val2017"

RECIPES = {
    "s8s8_minmax": (QuantType.QInt8, CalibrationMethod.MinMax, False),
    "s8s8_minmax_reduce_range": (QuantType.QInt8, CalibrationMethod.MinMax, True),
    "u8u8_minmax": (QuantType.QUInt8, CalibrationMethod.MinMax, False),
    "u8u8_percentile": (QuantType.QUInt8, CalibrationMethod.Percentile, False),
}
MODELS = [
    ("resnet50-tiny-imagenet", "tiny_imagenet", TINY, None, "accuracy"),
    ("resnet50", "imagenet_224", TINY, None, "agreement"),
    ("resnet50-embed", "imagenet_224", TINY, None, "cosine"),
    ("yolov8n", "yolo_640", COCO, ["Conv"], "detection"),
]


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def _held_out(directory: Path) -> list[Path]:
    files = sorted(f for f in directory.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})
    return files[1000:1500]


def main() -> int:
    root = find_dataset_root(REPO_ROOT / "data")
    labelled = TinyImageNetVal(root, TinyImageNetTrain(root).class_to_idx).samples[:2000]
    report: dict[str, object] = {"generated_at": datetime.now(UTC).isoformat(), "models": {}}

    with tempfile.TemporaryDirectory() as tmp:
        for name, preset, calibration_dir, op_types, metric in MODELS:
            cfg = PREPROCESS_PRESETS[preset]
            source = ARTIFACTS / f"{name}.onnx"
            prepared = Path(tmp) / f"{name}_prep.onnx"
            try:
                quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)
            except Exception:
                quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
            fp32 = _session(source)
            input_name = fp32.get_inputs()[0].name
            calibration = list(iter_calibration_images(calibration_dir, cfg, 100))

            if metric == "accuracy":
                items = [(preprocess(p.read_bytes(), cfg).array, y) for p, y in labelled]
            else:
                items = [
                    (preprocess(p.read_bytes(), cfg).array, None)
                    for p in _held_out(calibration_dir)
                ]
            reference = [_run(fp32, x) for x, _ in items]
            row: dict[str, object] = {"metric": metric, "images": len(items)}
            if metric == "accuracy":
                row["fp32"] = round(
                    float(
                        np.mean(
                            [r[0].argmax() == y for r, (_, y) in zip(reference, items, strict=True)]
                        )
                    ),
                    4,
                )

            for recipe, (qtype, method, reduce_range) in RECIPES.items():
                out = Path(tmp) / f"{name}_{recipe}.onnx"
                quantize_static(
                    str(prepared),
                    str(out),
                    _OnnxCalibrationReader(calibration, input_name),
                    quant_format=QuantFormat.QDQ,
                    activation_type=qtype,
                    weight_type=qtype,
                    per_channel=True,
                    reduce_range=reduce_range,
                    calibrate_method=method,
                    op_types_to_quantize=op_types,
                )
                session = _session(out)
                outputs = [_run(session, x) for x, _ in items]
                if metric == "accuracy":
                    score: object = np.mean(
                        [o[0].argmax() == y for o, (_, y) in zip(outputs, items, strict=True)]
                    )
                elif metric == "agreement":
                    score = np.mean(
                        [
                            o[0].argmax() == r[0].argmax()
                            for o, r in zip(outputs, reference, strict=True)
                        ]
                    )
                elif metric == "cosine":
                    cos = [
                        float(o.ravel() @ r.ravel() / (np.linalg.norm(o) * np.linalg.norm(r)))
                        for o, r in zip(outputs, reference, strict=True)
                    ]
                    score = {
                        "mean": round(float(np.mean(cos)), 4),
                        "min": round(float(np.min(cos)), 4),
                    }
                else:
                    score = np.mean(
                        [
                            _dominant_class(o) == _dominant_class(r)
                            for o, r in zip(outputs, reference, strict=True)
                        ]
                    )
                row[recipe] = score if isinstance(score, dict) else round(float(score), 4)
                print(f"{name:<24} {recipe:<26} {row[recipe]}", flush=True)
            report["models"][name] = row  # type: ignore[index]

    out = REPO_ROOT / "benchmarks" / "reports" / "int8_recipes.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
