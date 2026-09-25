"""INT8 quantization for ONNX and PyTorch models.

Plain English:
    A trained network stores its weights as 32-bit floating point numbers.
    Quantization rewrites them as 8-bit integers. Each number now takes a
    quarter of the space, and integer arithmetic runs faster than floating
    point on most CPUs.

    The catch: 8 bits can only represent 256 distinct values, so you lose
    precision. The whole craft of quantization is choosing the *scale factor*
    that maps each float range onto those 256 slots with the least damage.

Two approaches are implemented, and the difference matters:

* **Dynamic quantization** — weights are converted ahead of time; the scale
  for activations is computed on the fly, per batch. No calibration data
  needed, so it is trivial to apply. Best for models dominated by big matrix
  multiplies.
* **Static quantization** — activations are quantized too, using scales
  measured by running a few hundred real images through the model first
  ("calibration"). Faster than dynamic at inference time, and usually more
  accurate, but you must supply representative data. Feeding it random noise
  produces wrong scales and destroys accuracy, so this module insists on real
  calibration images for the static path.

Accuracy is always measured, never assumed: every function reports the
observed output difference against the original model.

Usage::

    python -m models.optimization.quantize --onnx models/artifacts/resnet18.onnx --mode dynamic
    python -m models.optimization.quantize --onnx models/artifacts/resnet18.onnx \
        --mode static --calibration-dir data/tiny-imagenet-200/val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.utils.image_processing import PreprocessConfig, preprocess


@dataclass
class QuantizationResult:
    """Outcome of one quantization run, including the accuracy cost."""

    name: str
    mode: str
    source_path: str
    output_path: str
    original_mb: float
    quantized_mb: float
    compression_ratio: float
    max_abs_diff: float
    mean_abs_diff: float
    top1_agreement: float
    duration_seconds: float
    calibration_images: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.name} [{self.mode}]: {self.original_mb:.1f} MB -> "
            f"{self.quantized_mb:.1f} MB ({self.compression_ratio:.2f}x smaller), "
            f"top-1 agreement {self.top1_agreement:.1%}, max diff {self.max_abs_diff:.3f}"
        )


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def iter_calibration_images(
    directory: Path, cfg: PreprocessConfig, limit: int = 200
) -> Iterator[np.ndarray]:
    """Yield preprocessed arrays from a directory tree of images.

    Walks recursively so a standard ``ImageFolder`` layout (one sub-directory
    per class) works without any extra arguments.
    """
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
    count = 0
    for path in sorted(directory.rglob("*")):
        if count >= limit:
            return
        if path.suffix not in suffixes or not path.is_file():
            continue
        try:
            yield preprocess(path.read_bytes(), cfg).array
            count += 1
        except Exception:  # noqa: S112 - see below
            # A corrupt calibration image is skipped deliberately and without
            # logging: calibration walks hundreds of files, and one unreadable
            # image is not worth a log line per occurrence. If NO images load,
            # quantize_onnx_static raises a clear error instead.
            continue


class _OnnxCalibrationReader:
    """Feeds calibration batches to ONNX Runtime's static quantizer.

    ONNX Runtime asks for one input dict at a time and expects ``None`` when
    the data is exhausted, so this wraps our generator in that protocol.
    """

    def __init__(self, arrays: list[np.ndarray], input_name: str) -> None:
        self._input_name = input_name
        self._iter = iter(arrays)

    def get_next(self) -> dict[str, np.ndarray] | None:
        item = next(self._iter, None)
        return None if item is None else {self._input_name: item}

    def rewind(self) -> None:
        raise NotImplementedError  # not used by the calibrator we call


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------
def _compare_or_note(
    original: Path, quantized: Path, samples: list[np.ndarray], notes: list[str]
) -> tuple[float, float, float]:
    """Compare the two models, or record why the comparison was impossible.

    Quantization and verification are separate steps, and only the first one
    produced the artifact. A quantized model that will not load is a real and
    useful finding - ONNX Runtime's CPU provider has no `ConvInteger` kernel,
    so dynamically quantizing a convolutional model yields exactly that - but
    letting the load error propagate discards the compression numbers too, and
    reports a crash where the honest answer is "smaller, unverifiable".
    """
    try:
        return _compare_onnx_models(original, quantized, samples)
    except Exception as exc:  # any load or run failure is reportable, not fatal
        notes.append(
            f"accuracy NOT verified: the quantized model could not be executed "
            f"({type(exc).__name__}). The file was still written. A common cause "
            f"is dynamic quantization of a convolutional model, which emits "
            f"ConvInteger - unsupported by the ONNX Runtime CPU provider. Use "
            f"static quantization for convolutional models."
        )
        return 0.0, 0.0, 0.0


def _compare_onnx_models(
    original: Path, quantized: Path, samples: list[np.ndarray]
) -> tuple[float, float, float]:
    """Run both models over the same inputs and measure the disagreement.

    Returns:
        ``(max_abs_diff, mean_abs_diff, top1_agreement)``. Top-1 agreement is
        the fraction of inputs where both models pick the same winning class —
        the number that actually predicts whether users notice the change.
    """
    import onnxruntime as ort

    sess_a = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
    name_a = sess_a.get_inputs()[0].name
    name_b = sess_b.get_inputs()[0].name

    max_diff = 0.0
    diffs: list[float] = []
    agree = 0
    total = 0

    for arr in samples:
        out_a = sess_a.run(None, {name_a: arr})[0]
        out_b = sess_b.run(None, {name_b: arr})[0]
        d = np.abs(out_a.astype(np.float64) - out_b.astype(np.float64))
        max_diff = max(max_diff, float(d.max()))
        diffs.append(float(d.mean()))
        if out_a.ndim == 2:
            agree += int(np.argmax(out_a, axis=1)[0] == np.argmax(out_b, axis=1)[0])
            total += 1

    return max_diff, float(np.mean(diffs)) if diffs else 0.0, (agree / total if total else 1.0)


# ---------------------------------------------------------------------------
# ONNX quantization
# ---------------------------------------------------------------------------


def _concrete_input_shape(model_path: Path) -> list[int]:
    """Work out a usable input shape for a model with dynamic axes.

    A dynamically-shaped ONNX model reports its spatial dimensions as strings
    ("height", "width") rather than integers. Substituting 1 for those — the
    obvious first guess — produces a 1x1 image, which crashes any network
    containing an upsample-and-concatenate block such as YOLO's neck.

    Batch is safe to set to 1. For the spatial dimensions we fall back to 640,
    the standard detector input size; callers that know better should pass
    ``input_shape`` explicitly.
    """
    import onnxruntime as ort

    shape = (
        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        .get_inputs()[0]
        .shape
    )

    concrete: list[int] = []
    for axis, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            concrete.append(dim)
        elif axis == 0:
            concrete.append(1)  # batch
        elif axis == 1:
            concrete.append(3)  # channels
        else:
            concrete.append(640)  # spatial
    return concrete


def quantize_onnx_dynamic(
    src: Path,
    dst: Path | None = None,
    *,
    samples: list[np.ndarray] | None = None,
    per_channel: bool = True,
    input_shape: tuple[int, ...] | None = None,
) -> QuantizationResult:
    """Dynamically quantize an ONNX model to INT8 weights.

    Args:
        src: Path to the float32 ``.onnx`` model.
        dst: Output path. Defaults to ``<src stem>_int8.onnx``.
        samples: Inputs used to measure the accuracy cost. Random noise is
            used if omitted, which measures numerical drift but not real
            accuracy — pass real images when you can.
        per_channel: Compute a separate scale per output channel rather than
            one for the whole tensor. Costs nothing at inference time and
            noticeably reduces error on convolutional models.

    Returns:
        :class:`QuantizationResult` with the measured size and accuracy impact.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(src)
    dst = Path(dst) if dst else src.with_name(f"{src.stem}_int8.onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        # Keep the graph's own optimisation pass out of this: ORT applies it
        # at load time anyway, and running it here has been a source of
        # shape-inference failures on exported torchvision graphs.
        extra_options={"EnableSubgraph": False},
    )
    duration = time.perf_counter() - started

    notes: list[str] = []
    if samples is None:
        concrete = list(input_shape) if input_shape else _concrete_input_shape(src)
        samples = [np.random.randn(*concrete).astype(np.float32) for _ in range(16)]
        notes.append(
            "accuracy measured on random noise; supply real images for a meaningful number"
        )

    max_diff, mean_diff, agreement = _compare_or_note(src, dst, samples, notes)

    original_mb = src.stat().st_size / 1_048_576
    quantized_mb = dst.stat().st_size / 1_048_576

    return QuantizationResult(
        name=src.stem,
        mode="onnx_dynamic",
        source_path=str(src),
        output_path=str(dst),
        original_mb=original_mb,
        quantized_mb=quantized_mb,
        compression_ratio=original_mb / quantized_mb if quantized_mb else 0.0,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        top1_agreement=agreement,
        duration_seconds=duration,
        notes=notes,
    )


def quantize_onnx_static(
    src: Path,
    calibration_dir: Path,
    cfg: PreprocessConfig,
    dst: Path | None = None,
    *,
    num_calibration: int = 200,
    per_channel: bool = True,
    trt_compatible: bool = False,
) -> QuantizationResult:
    """Statically quantize an ONNX model using real calibration images.

    Static quantization needs to know the typical *range* of the activations
    flowing through each layer. It learns those ranges by running real images
    through the model. Using unrepresentative data here is the single biggest
    cause of "quantization destroyed my accuracy", which is why this function
    requires a directory of real images and refuses to invent them.

    Set ``trt_compatible`` to produce a graph TensorRT can build an engine
    from. ONNX Runtime quantizes biases to INT32, which is correct - a bias
    scale is ``input_scale * weight_scale``, and int8 would overflow - and
    which TensorRT rejects, because its ``DequantizeLinear`` accepts only 8-
    and 4-bit types. It fails at the first bias node with *"input has type
    Int32 but must have type FP8, FP4, Int4, Int8, or UInt8"*. The flag leaves
    biases in fp32 instead; on resnet50-tiny-imagenet that drops 54 of 182 DQ
    nodes and changes the file size by under 1%.

    It writes to a separate file (``<name>_int8_trt.onnx``) on purpose. The
    CPU INT8 figures in ``benchmarks/reports/quantization.json`` were measured
    against ``<name>_int8_static.onnx``, and quietly changing what that name
    contains would invalidate them.

    Args:
        trt_compatible: Emit a graph TensorRT will parse, at the cost of
            leaving biases unquantized.

    Raises:
        FileNotFoundError: The calibration directory does not exist.
        ValueError: No usable calibration images were found.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    src = Path(src)
    calibration_dir = Path(calibration_dir)
    suffix = "_int8_trt" if trt_compatible else "_int8_static"
    dst = Path(dst) if dst else src.with_name(f"{src.stem}{suffix}.onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not calibration_dir.exists():
        raise FileNotFoundError(f"calibration directory not found: {calibration_dir}")

    import onnxruntime as ort

    input_name = (
        ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    )
    samples = list(iter_calibration_images(calibration_dir, cfg, num_calibration))
    if not samples:
        raise ValueError(
            f"no usable calibration images found under {calibration_dir}. "
            "Static quantization cannot proceed without representative data."
        )

    started = time.perf_counter()

    # Shape inference + graph clean-up. Static quantization fails on graphs
    # with unknown intermediate shapes, which exported models often have.
    #
    # Symbolic shape inference cannot always resolve a graph exported with
    # dynamic axes (YOLO's neck is a common example) and raises "Incomplete
    # symbolic shape inference". Retrying with it skipped lets ONNX Runtime
    # fall back to concrete shape inference from the calibration data, which
    # is enough for quantization even though it is less thorough.
    preprocessed = src.with_name(f"{src.stem}_prep.onnx")
    try:
        quant_pre_process(str(src), str(preprocessed), skip_symbolic_shape=False)
    except Exception:
        quant_pre_process(str(src), str(preprocessed), skip_symbolic_shape=True)

    quantize_static(
        model_input=str(preprocessed),
        model_output=str(dst),
        calibration_data_reader=_OnnxCalibrationReader(samples, input_name),
        quant_format=QuantFormat.QDQ,  # QDQ is the portable, widely supported form
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        # Empty rather than None when off: ORT treats the two the same, and an
        # explicit dict keeps the call one shape instead of two.
        extra_options={"QuantizeBias": False} if trt_compatible else {},
    )
    duration = time.perf_counter() - started
    preprocessed.unlink(missing_ok=True)

    notes: list[str] = []
    if trt_compatible:
        notes.append(
            "biases left in fp32 (QuantizeBias=False) so TensorRT will parse the graph; "
            "ONNX Runtime's default INT32 bias DequantizeLinear is rejected by its builder"
        )
    max_diff, mean_diff, agreement = _compare_or_note(src, dst, samples[:32], notes)

    original_mb = src.stat().st_size / 1_048_576
    quantized_mb = dst.stat().st_size / 1_048_576

    return QuantizationResult(
        name=src.stem,
        mode="onnx_static_trt" if trt_compatible else "onnx_static",
        source_path=str(src),
        output_path=str(dst),
        original_mb=original_mb,
        quantized_mb=quantized_mb,
        compression_ratio=original_mb / quantized_mb if quantized_mb else 0.0,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        top1_agreement=agreement,
        duration_seconds=duration,
        calibration_images=len(samples),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# PyTorch quantization
# ---------------------------------------------------------------------------
def quantize_torch_dynamic(
    model: Any,
    dst: Path,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    name: str = "model",
) -> QuantizationResult:
    """Dynamically quantize a PyTorch model and save it as TorchScript.

    Note the deliberate limitation: PyTorch's dynamic quantization only
    targets ``Linear`` (and RNN) layers, not ``Conv2d``. On a convolutional
    network like ResNet, almost all the weight is in convolutions, so the file
    barely shrinks. That is expected, and it is exactly why the ONNX path is
    the one we ship — it quantizes convolutions too.

    The result is saved with ``torch.jit.save`` rather than ``torch.save``
    because serving loads TorchScript, which does not execute pickled code.
    """
    import torch

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    example = torch.randn(*input_shape)

    with torch.inference_mode():
        baseline = model(example)

    started = time.perf_counter()
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    duration = time.perf_counter() - started

    with torch.inference_mode():
        after = quantized(example)
        scripted = torch.jit.trace(quantized, example)
        scripted = torch.jit.freeze(scripted)
    torch.jit.save(scripted, str(dst))

    # Save the float32 baseline too, so the size comparison is apples-to-apples.
    fp32_path = dst.with_name(f"{name}_fp32.pt")
    with torch.inference_mode():
        fp32_scripted = torch.jit.freeze(torch.jit.trace(model, example))
    torch.jit.save(fp32_scripted, str(fp32_path))

    diff = torch.abs(baseline - after)
    agreement = float((baseline.argmax(dim=1) == after.argmax(dim=1)).float().mean())

    original_mb = fp32_path.stat().st_size / 1_048_576
    quantized_mb = dst.stat().st_size / 1_048_576

    return QuantizationResult(
        name=name,
        mode="torch_dynamic",
        source_path=str(fp32_path),
        output_path=str(dst),
        original_mb=original_mb,
        quantized_mb=quantized_mb,
        compression_ratio=original_mb / quantized_mb if quantized_mb else 0.0,
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        top1_agreement=agreement,
        duration_seconds=duration,
        notes=[
            "PyTorch dynamic quantization covers Linear layers only; "
            "convolution weights stay float32, so CNN size reduction is small"
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize a model to INT8.")
    parser.add_argument("--onnx", type=Path, required=True, help="Source .onnx model.")
    parser.add_argument("--mode", choices=["dynamic", "static", "both"], default="dynamic")
    parser.add_argument(
        "--calibration-dir", type=Path, default=None, help="Images for static mode."
    )
    parser.add_argument(
        "--trt-compatible",
        action="store_true",
        help=(
            "With --mode static, also write <name>_int8_trt.onnx with biases left in "
            "fp32. TensorRT rejects the INT32 bias nodes in the default static graph."
        ),
    )
    parser.add_argument("--num-calibration", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "quantization.json",
    )
    args = parser.parse_args()

    if not args.onnx.is_file():
        print(f"error: no such model file: {args.onnx}", file=sys.stderr)
        return 2

    cfg = PreprocessConfig(size=(args.image_size, args.image_size))
    results: list[QuantizationResult] = []

    samples: list[np.ndarray] | None = None
    if args.calibration_dir and args.calibration_dir.exists():
        samples = list(iter_calibration_images(args.calibration_dir, cfg, 64))
        print(f"loaded {len(samples)} real images for accuracy comparison")

    if args.mode in ("dynamic", "both"):
        results.append(quantize_onnx_dynamic(args.onnx, samples=samples))

    if args.mode in ("static", "both"):
        # `is_dir()` rather than just "was it passed": a mistyped path used to
        # reach quantize_onnx_static and surface as a bare FileNotFoundError,
        # when the same helpful message applies.
        if not args.calibration_dir or not args.calibration_dir.is_dir():
            print(
                "error: --calibration-dir must name an existing directory of images.\n"
                "Static quantization needs real images to measure activation ranges.",
                file=sys.stderr,
            )
            return 2
        results.append(
            quantize_onnx_static(
                args.onnx, args.calibration_dir, cfg, num_calibration=args.num_calibration
            )
        )
        if args.trt_compatible:
            results.append(
                quantize_onnx_static(
                    args.onnx,
                    args.calibration_dir,
                    cfg,
                    num_calibration=args.num_calibration,
                    trt_compatible=True,
                )
            )

    for result in results:
        print(result.summary())
        for note in result.notes:
            print(f"  note: {note}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
