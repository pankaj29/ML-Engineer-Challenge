"""Check the committed embedding and detection ONNX files against PyTorch.

The export step records parity for the models it exported in the same run
(benchmarks/reports/onnx_export.json). This covers the other two by rebuilding
each PyTorch model the way scripts/prepare_models.py does and running the
three sample photos through both, with each model's serving preprocessing.

Needs torchvision (downloads the ImageNet weights) and ultralytics
(requirements-train.txt).

    python scripts/verify_onnx_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.services.model_service import PREPROCESS_PRESETS
from api.utils.image_processing import preprocess

ARTIFACTS = REPO_ROOT / "models" / "artifacts"
SAMPLES = sorted((REPO_ROOT / "samples").glob("*.jpg"))
REPORT = REPO_ROOT / "benchmarks" / "reports" / "onnx_export.json"


class _Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm

        self.base = tvm.resnet50(weights="DEFAULT")
        self.base.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        return features / features.norm(dim=1, keepdim=True).clamp(min=1e-12)


def _detector() -> nn.Module:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt").model
    model.eval()
    return model


def _compare(name: str, model: nn.Module, preset: str, tolerance: float) -> dict[str, object]:
    session = ort.InferenceSession(
        str(ARTIFACTS / f"{name}.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    diffs = []
    model.eval()
    for path in SAMPLES:
        x = preprocess(path.read_bytes(), PREPROCESS_PRESETS[preset]).array
        with torch.no_grad():
            out = model(torch.from_numpy(x))
        reference = (out[0] if isinstance(out, (tuple, list)) else out).numpy()
        exported = session.run(None, {input_name: x})[0]
        diffs.append(np.abs(reference.astype(np.float64) - exported.astype(np.float64)))
    max_diff = float(max(d.max() for d in diffs))
    return {
        "name": name,
        "input_shape": list(session.get_inputs()[0].shape),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(np.mean([d.mean() for d in diffs])),
        "verified": max_diff <= tolerance,
        "tolerance": tolerance,
        "notes": [
            f"checked after export against the committed file on {len(SAMPLES)} sample photos "
            "(scripts/verify_onnx_parity.py)"
        ],
    }


def main() -> int:
    results = [
        _compare("resnet50-embed", _Embedding(), "imagenet_224", 1e-4),
        # Box coordinates run to 640, so the absolute tolerance is looser.
        _compare("yolov8n", _detector(), "yolo_640", 1e-2),
    ]
    existing = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else []
    names = {r["name"] for r in results}
    merged = [r for r in existing if r.get("name") not in names] + results
    REPORT.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['name']:<16} max abs diff {r['max_abs_diff']:.2e}  verified={r['verified']}")
    print(f"wrote {REPORT}")
    return 0 if all(r["verified"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
