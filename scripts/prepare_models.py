"""Prepare all three serving models end to end.

Plain English:
    One command that leaves the system ready to serve. For each of the three
    tasks it downloads or loads a model, exports it to ONNX, quantizes it to
    INT8, writes out the class labels, and registers everything.

    Run this once after cloning; the API then has real models to serve.

The three models and why each was chosen:

* **Classification — ResNet-50.** A residual network is the honest default for
  image classification: well understood, strong accuracy, and fast enough on
  CPU to meet the sub-second requirement. It is also the architecture we
  fine-tune on Tiny-ImageNet, so the serving path and the training path
  exercise the same code.

* **Detection — YOLOv8n.** The "n" is nano: roughly 6 MB and 3.2 M parameters.
  Detection is far more expensive than classification, and a heavier detector
  would blow the latency budget on CPU. YOLOv8 also exports cleanly to ONNX,
  which several detection architectures do not.

* **Similarity — ResNet-50 with the classifier head removed.** Chopping off
  the final layer leaves the 2048-dimensional feature vector the network had
  built before deciding on a class. Those features are exactly what we want
  for "does this look like that". Reusing the classification backbone also
  means one download serves two tasks.

Usage::

    python scripts/prepare_models.py                 # all three
    python scripts/prepare_models.py --only classification
    python scripts/prepare_models.py --skip-quantization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS = REPO_ROOT / "models" / "artifacts"


def _force_utf8_stdout() -> None:
    """Make stdout UTF-8 safe (torch and ultralytics print emoji)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _quantize_best(
    onnx_path: Path, preprocess_name: str, *, op_types: list[str] | None = None
) -> tuple[str, str] | None:
    """Quantize a model, preferring static over dynamic quantization.

    Why the preference matters (measured on this project's hardware, an Intel
    Core Ultra 7 155H, with ResNet-50 at batch 1):

        float32 ONNX          75.7 ms    97.4 MB
        INT8 dynamic        1008.0 ms    24.5 MB   <- 13x SLOWER
        INT8 static QDQ      104.6 ms    24.9 MB

    Dynamic quantization recomputes activation scales on every call and falls
    back to poorly-optimised integer convolution kernels, which is
    catastrophic for a convolutional network. Static quantization measures
    those scales once, ahead of time, from real calibration images — and is
    roughly 10x faster as a result.

    Static therefore requires calibration data. When none is available we fall
    back to dynamic and say so, rather than silently shipping the slow variant
    as if it were an optimisation.

    Returns:
        ``(artifact_key, filename)`` to register, or None if quantization failed.
    """
    from api.services.model_service import PREPROCESS_PRESETS
    from models.optimization.quantize import quantize_onnx_dynamic, quantize_onnx_static

    calibration_dir = _find_calibration_images(detection=preprocess_name == "yolo_640")

    if calibration_dir is not None:
        try:
            # Calibrate on exactly what the model sees when serving. A plain
            # PreprocessConfig here fed YOLO ImageNet-normalised center crops
            # instead of 0-1 letterboxed frames.
            cfg = PREPROCESS_PRESETS[preprocess_name]
            result = quantize_onnx_static(
                onnx_path, calibration_dir, cfg, num_calibration=100, op_types_to_quantize=op_types
            )
            print(
                f"int8       : {result.quantized_mb:.1f} MB "
                f"({result.compression_ratio:.2f}x smaller), static QDQ from "
                f"{result.calibration_images} real images, "
                f"top-1 agreement {result.top1_agreement:.1%}"
            )
            return "onnx_int8", Path(result.output_path).name
        except Exception as exc:
            print(
                f"int8       : static quantization failed ({type(exc).__name__}: "
                f"{str(exc)[:120]}); falling back to dynamic",
                file=sys.stderr,
            )

    try:
        result = quantize_onnx_dynamic(onnx_path)
        print(
            f"int8       : {result.quantized_mb:.1f} MB "
            f"({result.compression_ratio:.2f}x smaller), DYNAMIC - expect it to be "
            "slower than float32 on CPU; supply calibration data for the static path"
        )
        return "onnx_int8", Path(result.output_path).name
    except Exception as exc:
        print(f"int8       : skipped ({type(exc).__name__}: {str(exc)[:120]})", file=sys.stderr)
        return None


def _find_calibration_images(*, detection: bool = False) -> Path | None:
    """Locate a directory of real images to calibrate static quantization.

    Static quantization needs images that resemble production traffic. The
    Tiny-ImageNet validation split is used when present because it is already
    downloaded for training. A detector prefers COCO val2017
    (``download_datasets.py --dataset coco_sample``): upscaled 64px thumbnails
    contain nothing it detects confidently, so the class-logit range it
    calibrates is too narrow and every INT8 score saturates near 0.5.
    """
    coco = [REPO_ROOT / "data" / "coco_val2017" / "val2017", REPO_ROOT / "data" / "coco_val2017"]
    candidates = (coco if detection else []) + [
        REPO_ROOT / "data" / "tiny-imagenet-200" / "tiny-imagenet-200" / "val" / "images",
        REPO_ROOT / "data" / "tiny-imagenet-200" / "val" / "images",
        REPO_ROOT / "data" / "calibration",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def prepare_classification(
    *, quantize: bool = True, arch: str = "resnet50", overwrite: bool = False
) -> dict[str, Any]:
    """Export a pretrained ImageNet classifier and register it."""
    import torchvision.models as tvm

    from models.optimization.export_onnx import export_to_onnx
    from models.registry import Registry

    print(f"\n=== Classification: {arch} ===")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    model = getattr(tvm, arch)(weights="DEFAULT")

    # The label list ships with the torchvision weights, so serving returns
    # "golden retriever" rather than "class_207". `get_model_weights` is used
    # rather than guessing the enum's name: torchvision's casing is irregular
    # (ResNet50_Weights, EfficientNet_B0_Weights) and a wrong guess silently
    # yields no labels at all.
    labels: list[str] = []
    try:
        labels = list(tvm.get_model_weights(arch).DEFAULT.meta["categories"])
    except (KeyError, AttributeError, ValueError) as exc:
        print(f"warning    : could not read class labels for {arch}: {exc}", file=sys.stderr)
    labels_path = ARTIFACTS / "imagenet_labels.json"
    if labels:
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        print(f"labels     : {len(labels)} classes -> {labels_path.name}")

    onnx_path = ARTIFACTS / f"{arch}.onnx"
    result = export_to_onnx(model, onnx_path, input_shape=(1, 3, 224, 224), name=arch)
    print(
        f"onnx       : {result.size_mb:.1f} MB, max diff {result.max_abs_diff:.2e}, verified={result.verified}"
    )

    artifacts = {"onnx": onnx_path.name}

    if quantize:
        quantized = _quantize_best(onnx_path, "imagenet_224")
        if quantized:
            artifacts[quantized[0]] = quantized[1]

    Registry().register(
        name=arch,
        version="1.0.0",
        task="classification",
        artifacts=artifacts,
        preprocess="imagenet_224",
        labels_file=labels_path.name if labels else None,
        num_classes=len(labels) or 1000,
        input_shape=[1, 3, 224, 224],
        description=(
            f"{arch} pretrained on ImageNet-1k, exported to ONNX. "
            "Serves general-purpose image classification over 1000 categories."
        ),
        limitations=[
            "Trained on ImageNet-1k: only recognises those 1000 categories, and "
            "will confidently return a wrong label for anything outside them.",
            "ImageNet is known to under-represent non-Western contexts, so accuracy "
            "is uneven across geographies and cultures.",
            "Expects a single dominant subject; cluttered scenes with several "
            "objects are better served by the detection endpoint.",
            "Confidence is not calibrated — a 0.9 score does not mean 90% correct.",
        ],
        is_default=True,
        overwrite=True,
    )
    print(f"registered : {arch}:1.0.0 (classification, default)")
    return {"model": f"{arch}:1.0.0", "artifacts": artifacts}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def prepare_detection(*, quantize: bool = True, model_name: str = "yolov8n") -> dict[str, Any]:
    """Download YOLOv8, export to ONNX and register it."""
    from models.registry import Registry

    print(f"\n=== Detection: {model_name} ===")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed; skipping detection.\n"
            "  install with: pip install -r requirements-train.txt",
            file=sys.stderr,
        )
        return {"skipped": "ultralytics not installed"}

    weights = ARTIFACTS / f"{model_name}.pt"
    yolo = YOLO(str(weights) if weights.exists() else f"{model_name}.pt")

    # Ultralytics writes the .onnx next to the .pt it loaded, so export and
    # then move the result into the artifacts directory.
    # dynamic=True marks the batch axis as variable. With dynamic=False the
    # graph is locked to batch 1 and the /batch endpoint crashes on any
    # multi-image request - caught by the batch_invariance validation check.
    exported = yolo.export(format="onnx", imgsz=640, simplify=False, dynamic=True, opset=17)
    exported_path = Path(exported)
    onnx_path = ARTIFACTS / f"{model_name}.onnx"
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(exported_path.read_bytes())
        exported_path.unlink(missing_ok=True)
    print(f"onnx       : {onnx_path.stat().st_size / 1_048_576:.1f} MB")

    # COCO class names, in the index order the model outputs.
    names = yolo.names
    labels = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    labels_path = ARTIFACTS / "coco_labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    print(f"labels     : {len(labels)} classes -> {labels_path.name}")

    artifacts = {"onnx": onnx_path.name}

    if quantize:
        # Conv only: see quantize_onnx_static for why the head must stay fp32.
        quantized = _quantize_best(onnx_path, "yolo_640", op_types=["Conv"])
        if quantized:
            artifacts[quantized[0]] = quantized[1]

    Registry().register(
        name=model_name,
        version="1.0.0",
        task="detection",
        artifacts=artifacts,
        preprocess="yolo_640",
        labels_file=labels_path.name,
        num_classes=len(labels),
        input_shape=[1, 3, 640, 640],
        description=(
            f"{model_name} (YOLOv8 nano) trained on COCO, exported to ONNX. "
            "Detects 80 common object categories with bounding boxes."
        ),
        limitations=[
            "Trained on COCO's 80 categories only; anything else is either "
            "missed entirely or mislabelled as the nearest COCO class.",
            "The nano variant trades accuracy for speed. Small, distant or "
            "heavily occluded objects are frequently missed.",
            "Fixed 640x640 input: very wide or very tall images lose detail "
            "after letterboxing, which hurts recall on small objects.",
            "Crowded scenes suffer from non-maximum suppression merging "
            "genuinely separate, overlapping objects into one box.",
        ],
        is_default=True,
        overwrite=True,
    )
    print(f"registered : {model_name}:1.0.0 (detection, default)")
    return {"model": f"{model_name}:1.0.0", "artifacts": artifacts}


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def prepare_similarity(*, quantize: bool = True, arch: str = "resnet50") -> dict[str, Any]:
    """Build an embedding model by removing the classifier head."""
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    from models.optimization.export_onnx import export_to_onnx
    from models.registry import Registry

    print(f"\n=== Similarity: {arch}-embeddings ===")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    backbone = getattr(tvm, arch)(weights="DEFAULT")
    feature_dim = backbone.fc.in_features

    # Replacing the final classification layer with Identity turns the network
    # into a feature extractor: the forward pass now returns the 2048-number
    # description the model had built, instead of 1000 class scores.
    backbone.fc = nn.Identity()

    class EmbeddingModel(nn.Module):
        """Wraps a backbone so it emits L2-normalised embeddings.

        Normalising inside the graph rather than in Python means the ONNX
        artifact is self-contained: whoever runs it gets unit vectors without
        having to remember an extra step.
        """

        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.base(x)
            return features / features.norm(dim=1, keepdim=True).clamp(min=1e-12)

    model = EmbeddingModel(backbone)

    name = f"{arch}-embed"
    onnx_path = ARTIFACTS / f"{name}.onnx"
    result = export_to_onnx(
        model, onnx_path, input_shape=(1, 3, 224, 224), name=name, output_names=["embedding"]
    )
    print(
        f"onnx       : {result.size_mb:.1f} MB, dim={feature_dim}, "
        f"max diff {result.max_abs_diff:.2e}, verified={result.verified}"
    )

    artifacts = {"onnx": onnx_path.name}

    if quantize:
        quantized = _quantize_best(onnx_path, "imagenet_224")
        if quantized:
            artifacts[quantized[0]] = quantized[1]

    Registry().register(
        name=name,
        version="1.0.0",
        task="similarity",
        artifacts=artifacts,
        preprocess="imagenet_224",
        num_classes=None,
        input_shape=[1, 3, 224, 224],
        description=(
            f"{arch} backbone with the classification head removed, emitting "
            f"{feature_dim}-dimensional L2-normalised embeddings for similarity search."
        ),
        limitations=[
            "Features come from an ImageNet classifier, so they encode "
            "'what object is this' far better than style, colour or composition.",
            "Not trained with a contrastive objective, so it is weaker at "
            "instance-level retrieval (finding the same specific object) than "
            "a purpose-built model such as CLIP or DINOv2.",
            "Similarity scores are only comparable within one index built by "
            "one model version; re-embedding is required after a model change.",
            "Search is exact brute force, which is linear in index size. "
            "Beyond ~1M vectors, switch to an approximate index.",
        ],
        is_default=True,
        overwrite=True,
    )
    print(f"registered : {name}:1.0.0 (similarity, default)")
    return {"model": f"{name}:1.0.0", "dimension": feature_dim, "artifacts": artifacts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare all serving models.")
    parser.add_argument(
        "--only",
        choices=["classification", "detection", "similarity"],
        action="append",
        help="Prepare only these tasks. Repeatable.",
    )
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--arch", default="resnet50", help="Classification/embedding backbone.")
    parser.add_argument("--detector", default="yolov8n")
    args = parser.parse_args()

    _force_utf8_stdout()

    wanted = set(args.only or ["classification", "detection", "similarity"])
    quantize = not args.skip_quantization
    summary: dict[str, Any] = {}

    if "classification" in wanted:
        summary["classification"] = prepare_classification(quantize=quantize, arch=args.arch)
    if "detection" in wanted:
        summary["detection"] = prepare_detection(quantize=quantize, model_name=args.detector)
    if "similarity" in wanted:
        summary["similarity"] = prepare_similarity(quantize=quantize, arch=args.arch)

    from models.registry import Registry

    problems = Registry().validate()
    print("\n=== Registry validation ===")
    if problems:
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        return 1
    print("  all registered artifacts exist")

    print("\nDone. Start the API with: uvicorn api.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
