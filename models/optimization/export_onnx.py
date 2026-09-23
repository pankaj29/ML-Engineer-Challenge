"""Export PyTorch models to ONNX.

Plain English:
    A PyTorch model is Python code. Every forward pass goes through the Python
    interpreter, which is slow and awkward to deploy. ONNX ("Open Neural
    Network Exchange") is a portable *graph* format: the same maths, written
    down as data instead of code. ONNX Runtime then executes that graph in
    C++, fusing operations together and skipping Python entirely.

    Typical result on CPU: 2-4x faster, with no change in accuracy.

Two details matter and are handled explicitly below:

* **Dynamic batch axis.** By default, tracing bakes in the batch size used
  during export, so a model exported with batch 1 crashes on a batch of 8.
  We mark the batch dimension dynamic so one artifact serves both.
* **Numerical verification.** Export can silently change results (an
  unsupported operator gets approximated). We always run the original and the
  exported graph on the same input and compare, rather than assuming success.

Usage::

    python -m models.optimization.export_onnx --model resnet50 --output models/artifacts
    python -m models.optimization.export_onnx --checkpoint models/artifacts/tiny.pt --num-classes 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _force_utf8_stdout() -> None:
    """Make stdout UTF-8 safe.

    torch.onnx's exporter prints progress with emoji. On a Windows console
    (default code page 1252) that raises UnicodeEncodeError and kills the
    export. Reconfiguring to UTF-8 with replacement makes the scripts behave
    identically on Windows, Linux and inside Docker.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
            pass


@dataclass
class ExportResult:
    """What happened during one export."""

    name: str
    onnx_path: Path
    input_shape: tuple[int, ...]
    opset: int
    size_mb: float
    max_abs_diff: float
    mean_abs_diff: float
    verified: bool
    export_seconds: float
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "onnx_path": str(self.onnx_path),
            "input_shape": list(self.input_shape),
            "opset": self.opset,
            "size_mb": round(self.size_mb, 2),
            "max_abs_diff": float(self.max_abs_diff),
            "mean_abs_diff": float(self.mean_abs_diff),
            "verified": self.verified,
            "export_seconds": round(self.export_seconds, 2),
            "output_shapes": [list(s) for s in self.output_shapes],
            "notes": self.notes,
        }


def export_to_onnx(
    model: Any,
    output_path: Path,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    opset: int = 17,
    name: str = "model",
    dynamic_batch: bool = True,
    tolerance: float = 1e-3,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> ExportResult:
    """Trace a PyTorch model into an ONNX graph and verify it numerically.

    Args:
        model: An ``nn.Module`` in any mode; it is switched to ``eval()``.
        output_path: Where to write the ``.onnx`` file.
        input_shape: Example input shape used for tracing.
        opset: ONNX operator-set version. 17 is broadly supported by ONNX
            Runtime 1.15+ and covers everything these models use.
        dynamic_batch: Mark axis 0 as variable so one file serves any batch size.
        tolerance: Maximum acceptable absolute difference between PyTorch and
            ONNX outputs. 1e-3 allows for float32 reassociation during graph
            fusion while still catching real correctness bugs.

    Returns:
        :class:`ExportResult` including the measured numerical difference.
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    dummy = torch.randn(*input_shape)
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {n: {0: "batch"} for n in [*input_names, *output_names]}

    started = time.perf_counter()
    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,  # pre-compute constant sub-graphs at export time
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }
    # torch >= 2.6 defaults to the new "dynamo" exporter. We explicitly opt out
    # of it for two concrete reasons, both caught by the checks further down:
    #   1. It ignores `dynamic_axes`, baking in the tracing batch size, so the
    #      exported graph crashes on any batch other than 1.
    #   2. It writes weights to a sidecar `.onnx.data` file, turning a single
    #      self-contained artifact into two files that must travel together.
    # The legacy TorchScript exporter does the right thing on both counts.
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy, str(output_path), **export_kwargs)
    export_seconds = time.perf_counter() - started

    sidecar = output_path.with_suffix(output_path.suffix + ".data")
    if sidecar.exists():
        notes_external = (
            f"weights were written externally to {sidecar.name}; "
            "the artifact is not self-contained"
        )
    else:
        notes_external = ""

    notes: list[str] = [notes_external] if notes_external else []

    # --- Structural check: is the graph even well-formed? ------------------
    try:
        import onnx

        graph = onnx.load(str(output_path))
        onnx.checker.check_model(graph)
    except ImportError:
        notes.append("onnx package not installed; skipped structural check")
    except Exception as exc:
        notes.append(f"structural check failed: {type(exc).__name__}: {exc}")

    # --- Numerical check: same input, same output? -------------------------
    # This is the check that actually matters. An export that produces a valid
    # graph computing the wrong thing is worse than one that fails loudly.
    import onnxruntime as ort

    with torch.inference_mode():
        torch_out = model(dummy)
    torch_arrays = (
        [t.cpu().numpy() for t in torch_out]
        if isinstance(torch_out, (list, tuple))
        else [torch_out.cpu().numpy()]
    )

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_arrays = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})

    diffs = [
        np.abs(t.astype(np.float64) - o.astype(np.float64))
        for t, o in zip(torch_arrays, onnx_arrays, strict=False)
    ]
    max_diff = float(max(d.max() for d in diffs)) if diffs else 0.0
    mean_diff = float(np.mean([d.mean() for d in diffs])) if diffs else 0.0
    verified = max_diff <= tolerance
    if not verified:
        notes.append(
            f"NUMERICAL MISMATCH: max abs diff {max_diff:.2e} exceeds tolerance {tolerance:.0e}"
        )

    # --- Dynamic batch check ----------------------------------------------
    if dynamic_batch:
        try:
            batched = np.random.randn(4, *input_shape[1:]).astype(np.float32)
            session.run(None, {session.get_inputs()[0].name: batched})
        except Exception as exc:
            notes.append(f"dynamic batch verification failed: {exc}")

    return ExportResult(
        name=name,
        onnx_path=output_path,
        input_shape=input_shape,
        opset=opset,
        size_mb=output_path.stat().st_size / 1_048_576,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        verified=verified,
        export_seconds=export_seconds,
        output_shapes=[tuple(a.shape) for a in onnx_arrays],
        notes=notes,
    )


def optimize_onnx_graph(src: Path, dst: Path) -> dict[str, Any]:
    """Run ONNX Runtime's offline graph optimisations and save the result.

    Doing this ahead of time means each serving process does not repeat the
    optimisation work on every startup.
    """
    import onnxruntime as ort

    dst.parent.mkdir(parents=True, exist_ok=True)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])

    return {
        "source_mb": round(src.stat().st_size / 1_048_576, 2),
        "optimized_mb": round(dst.stat().st_size / 1_048_576, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a torchvision or checkpointed model to ONNX.",
    )
    parser.add_argument(
        "--model",
        default="resnet50",
        help="torchvision model name to export with pretrained weights (default: resnet50).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional fine-tuned state_dict to load into the architecture before export.",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None, help="Override the classifier head size."
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Square input size for tracing."
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "artifacts")
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "onnx_export.json"
    )
    args = parser.parse_args()

    _force_utf8_stdout()

    import torch
    import torchvision.models as tvm

    if not hasattr(tvm, args.model):
        print(f"error: torchvision has no model named {args.model!r}", file=sys.stderr)
        return 2

    weights = None if args.checkpoint else "DEFAULT"
    model = getattr(tvm, args.model)(weights=weights)

    if args.num_classes:
        # Swap the classification head for one with the right class count.
        if hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
        elif hasattr(model, "classifier"):
            last = model.classifier[-1]
            model.classifier[-1] = torch.nn.Linear(last.in_features, args.num_classes)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        print(f"loaded checkpoint {args.checkpoint}")

    out_path = args.output / f"{args.model}.onnx"
    result = export_to_onnx(
        model,
        out_path,
        input_shape=(1, 3, args.image_size, args.image_size),
        opset=args.opset,
        name=args.model,
    )

    print(f"exported  : {result.onnx_path}")
    print(f"size      : {result.size_mb:.2f} MB")
    print(f"max diff  : {result.max_abs_diff:.2e}")
    print(f"verified  : {result.verified}")
    for note in result.notes:
        print(f"note      : {note}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"report    : {args.report}")

    return 0 if result.verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
