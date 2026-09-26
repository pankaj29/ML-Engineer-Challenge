"""Compare INT8 calibration methods on the fine-tuned classifier.

TensorRT only accepts symmetric quantization. This measures what that
constraint does to each calibration method: calibrate on 200 Tiny-ImageNet
validation images, then score top-1 agreement with fp32 on a disjoint 200.
The result is why the TensorRT graph uses percentile calibration.

    python scripts/compare_int8_calibration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.services.model_service import PREPROCESS_PRESETS
from models.optimization.quantize import _OnnxCalibrationReader, iter_calibration_images

MODEL = REPO_ROOT / "models" / "artifacts" / "resnet50-tiny-imagenet.onnx"
SAMPLES = 200

CONFIGS = [
    ("MinMax", "asymmetric", CalibrationMethod.MinMax, False),
    ("MinMax", "symmetric", CalibrationMethod.MinMax, True),
    ("Entropy", "symmetric", CalibrationMethod.Entropy, True),
    ("Percentile", "symmetric", CalibrationMethod.Percentile, True),
]


def _top1(path: Path, images: list[np.ndarray]) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.array([int(session.run(None, {name: x})[0].argmax()) for x in images])


def main() -> int:
    val = next(
        p
        for p in (
            REPO_ROOT / "data" / "tiny-imagenet-200" / "tiny-imagenet-200" / "val" / "images",
            REPO_ROOT / "data" / "tiny-imagenet-200" / "val" / "images",
        )
        if p.is_dir()
    )
    images = list(iter_calibration_images(val, PREPROCESS_PRESETS["tiny_imagenet"], SAMPLES * 2))
    calibration, held_out = images[:SAMPLES], images[SAMPLES:]
    reference = _top1(MODEL, held_out)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prep.onnx"
        quant_pre_process(str(MODEL), str(prepared), skip_symbolic_shape=False)
        for method, mode, calibrate, symmetric in CONFIGS:
            out = Path(tmp) / f"{method}_{mode}.onnx"
            started = time.perf_counter()
            quantize_static(
                str(prepared),
                str(out),
                _OnnxCalibrationReader(calibration, "input"),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=calibrate,
                extra_options={"ActivationSymmetric": symmetric, "WeightSymmetric": symmetric},
            )
            seconds = time.perf_counter() - started
            agreement = float((_top1(out, held_out) == reference).mean())
            results.append(
                {
                    "calibration": method,
                    "quantization": mode,
                    "top1_agreement": round(agreement, 4),
                    "calibration_seconds": round(seconds, 1),
                    "tensorrt_accepts": symmetric,
                }
            )
            print(f"{method:<11} {mode:<11} agreement {agreement:6.1%}  {seconds:5.1f}s")

    report = {
        "model": "resnet50-tiny-imagenet",
        "calibration_images": len(calibration),
        "held_out_images": len(held_out),
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    out = REPO_ROOT / "benchmarks" / "reports" / "int8_calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
