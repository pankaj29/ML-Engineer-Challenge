"""TensorRT engine export.

Plain English:
    TensorRT is NVIDIA's inference compiler. Where ONNX Runtime executes a
    portable graph, TensorRT *compiles* that graph specifically for the GPU in
    front of it: it fuses layers together, picks the fastest kernel for each
    operation by actually timing several candidates, and can run in fp16 or
    INT8. The result is typically 2-5x faster than ONNX Runtime on the same
    GPU.

    The price is portability. A TensorRT engine is built for one GPU
    architecture and one TensorRT version. An engine built on an A100 will not
    load on a T4, and one built with TensorRT 10.6 will not load under 10.7.
    You therefore build the engine **on the machine that will serve it**,
    usually as part of the container's first start rather than at image build
    time.

**Status: executed on an NVIDIA A100 (TensorRT 11.3).** An fp32 engine for
the fine-tuned Tiny-ImageNet classifier builds and verifies against ONNX.

The API has moved twice, and this module handles all three eras by probing
for attributes rather than parsing version strings:

* **8.x / 9.x** - ``EXPLICIT_BATCH`` network flag required; ``BuilderFlag.FP16``
  and ``.INT8`` select precision.
* **10.x** - explicit batch became the only mode and its flag was removed;
  the precision flags remain.
* **11.x** - the precision flags are gone too. Networks are STRONGLY_TYPED and
  precision comes from the dtypes in the ONNX graph, so a reduced-precision
  engine needs an ONNX file already in that precision. This module supplies
  one either way: fp16 by converting the graph first, INT8 by building from
  the QDQ graph that ``quantize.py --mode static`` writes as
  ``<name>_int8_static.onnx``. Ask for INT8 from a plain fp32 graph and it
  raises :class:`UnsupportedPrecisionError` rather than quietly building fp32
  and labelling it INT8.

Usage (on a GPU host)::

    pip install -r requirements-gpu.txt
    python -m models.optimization.export_tensorrt --onnx models/artifacts/resnet50.onnx --fp16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class UnsupportedPrecisionError(RuntimeError):
    """The installed TensorRT cannot build this precision from this ONNX file.

    Distinct from a build failure: nothing is wrong with the model or the GPU.
    TensorRT 11 removed the FP16/INT8 builder flags in favour of
    strongly-typed networks, so a reduced-precision engine now requires an
    ONNX file already in that precision. Callers catch this to record a skip
    rather than an error.
    """


@dataclass
class TensorRTExportResult:
    """Outcome of building one engine."""

    name: str
    engine_path: str
    precision: str
    onnx_mb: float
    engine_mb: float
    build_seconds: float
    max_batch_size: int
    gpu_name: str
    tensorrt_version: str
    max_abs_diff: float = 0.0
    verified: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.name} [{self.precision}]: {self.onnx_mb:.1f} MB ONNX -> "
            f"{self.engine_mb:.1f} MB engine, built in {self.build_seconds:.0f}s "
            f"on {self.gpu_name}, max diff {self.max_abs_diff:.2e}"
        )


def tensorrt_available() -> tuple[bool, str]:
    """Check whether TensorRT can be used on this machine.

    Returns:
        ``(available, reason)``. The reason explains what is missing, so the
        caller can print something more useful than "not available".
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "no CUDA device is visible to PyTorch"
    except ImportError:
        return False, "PyTorch is not installed"

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        return False, "the tensorrt package is not installed (pip install -r requirements-gpu.txt)"

    try:
        import pycuda.driver  # noqa: F401
    except ImportError:
        return False, "the pycuda package is not installed (pip install -r requirements-gpu.txt)"

    return True, "available"


class _EntropyCalibrator:
    """INT8 calibrator that feeds real images to TensorRT.

    TensorRT's INT8 mode needs to learn the typical range of every
    activation, exactly like ONNX Runtime's static quantization. It asks this
    object for batches until it returns ``None``.

    The calibration cache is written to disk: building it is slow, and reusing
    it makes subsequent engine builds much faster.
    """

    def __init__(self, batches: list[np.ndarray], cache_file: Path) -> None:
        import pycuda.driver as cuda
        import tensorrt as trt

        # Subclassing is done dynamically so that merely importing this module
        # on a CPU-only host does not require the tensorrt package.
        self._trt = trt
        self._cuda = cuda
        self._batches = iter(batches)
        self._cache_file = Path(cache_file)
        self._device_input: Any = None
        self._batch_shape = batches[0].shape if batches else None

    def get_batch_size(self) -> int:
        return int(self._batch_shape[0]) if self._batch_shape else 1

    def get_batch(self, names: list[str]) -> list[int] | None:
        batch = next(self._batches, None)
        if batch is None:
            return None
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if self._device_input is None:
            self._device_input = self._cuda.mem_alloc(batch.nbytes)
        self._cuda.memcpy_htod(self._device_input, batch)
        return [int(self._device_input)]

    def read_calibration_cache(self) -> bytes | None:
        if self._cache_file.exists():
            return self._cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_bytes(cache)


def convert_onnx_to_fp16(src: Path, dst: Path | None = None) -> Path:
    """Rewrite an ONNX graph's weights and compute in half precision.

    Why this exists: TensorRT 11 removed ``BuilderFlag.FP16``. Networks are
    strongly typed, so an fp16 engine can only come from an fp16 graph. This
    produces that graph.

    ``keep_io_types=True`` leaves the model's inputs and outputs as fp32 and
    inserts casts at the boundary. That matters for two reasons: every caller
    (benchmark, verification, the serving runtime) keeps feeding fp32 arrays
    with no special-casing, and the numerical comparison against the original
    fp32 model stays like-for-like. The interior - where all the compute time
    goes - runs in fp16.

    Returns:
        Path to the fp16 model. Written next to the source as
        ``<name>.fp16.onnx`` unless ``dst`` says otherwise.
    """
    # onnxruntime ships this converter, so fp16 needs no extra dependency.
    #
    # The obvious package for this job, onnxconverter-common, is deliberately
    # NOT used: it hard-pins protobuf==3.20.2, which drags protobuf below the
    # >=6.31.1 that onnx requires and sends pip back to building onnx from
    # source - a failure that already cost this project hours.
    # onnxruntime.transformers.float16 is the same algorithm with the same
    # keep_io_types option and no dependency cost.
    try:
        import onnx
        from onnxruntime.transformers import float16
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise UnsupportedPrecisionError(
            "fp16 conversion needs onnx and onnxruntime, both already in "
            "requirements.txt. Install them and retry."
        ) from exc

    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".fp16.onnx")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        # Converting is not free, and the result is deterministic.
        return dst

    model = onnx.load(str(src))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(dst))
    return dst


class MissingLFSContentError(RuntimeError):
    """The file is a Git LFS pointer, so the real model was never fetched."""


def reject_lfs_pointer(path: Path) -> None:
    """Fail with an explanation if `path` is an LFS pointer rather than a model.

    ``models/artifacts/*.onnx`` is LFS-tracked. A clone on a runtime without
    git-lfs configured succeeds and looks completely normal, but every one of
    those files is a ~130 byte text pointer. Handing one to the ONNX parser
    gets you ``DecodeError: Error parsing message with type
    'onnx.ModelProto'`` - which is true, unhelpful, and mentions neither LFS
    nor the file. Check first and name the real problem.
    """
    try:
        with Path(path).open("rb") as handle:
            head = handle.read(42)
    except OSError:
        return

    if not head.startswith(b"version https://git-lfs"):
        return

    raise MissingLFSContentError(
        f"{path} is a Git LFS pointer of {path.stat().st_size} bytes, not a model. "
        "This checkout has the pointers but not the content. Fetch it with:\n"
        "    git lfs install && git lfs pull\n"
        "If git-lfs itself is missing (a bare Colab or CI runtime), install it "
        "first: apt-get install -y git-lfs"
    )


def check_trt_qdq_graph(onnx_path: Path) -> list[str]:
    """List the reasons TensorRT would refuse this QDQ graph. Empty means OK.

    TensorRT is stricter than ONNX Runtime about what a QDQ graph may contain,
    and its parser stops at the first offending node - so a graph with two
    problems takes two builds to diagnose, each needing a GPU. Both known
    constraints are properties of the file:

    * **Every DequantizeLinear input must be 8- or 4-bit.** ONNX Runtime
      quantizes biases to INT32, which TensorRT rejects with *"input has type
      Int32 but must have type FP8, FP4, Int4, Int8, or UInt8"*.
    * **Every zero point must be zero.** TensorRT supports only symmetric
      quantization and fails with *"Non-zero zero point is not supported"*.

    ``quantize.py --trt-compatible`` produces a graph that satisfies both.

    This is a pre-flight, not a guarantee: it encodes the constraints we have
    actually hit, and a build can still fail on something not listed here.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    eight_bit = {
        TensorProto.INT8,
        TensorProto.UINT8,
        TensorProto.INT4,
        TensorProto.UINT4,
        getattr(TensorProto, "FLOAT8E4M3FN", -1),
    }
    type_name = {v: k for k, v in TensorProto.DataType.items()}

    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    dtype = {init.name: init.data_type for init in graph.initializer}
    for info in list(graph.value_info) + list(graph.input):
        dtype[info.name] = info.type.tensor_type.elem_type
    constants = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    wrong_dtype: list[tuple[str, str]] = []
    nonzero_zp: list[str] = []
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if node.op_type == "DequantizeLinear":
            found = dtype.get(node.input[0])
            if found is not None and found not in eight_bit:
                wrong_dtype.append((node.name, type_name.get(found, str(found))))
        if len(node.input) > 2:
            zero_point = constants.get(node.input[2])
            if zero_point is not None and bool(np.any(np.asarray(zero_point) != 0)):
                nonzero_zp.append(node.name)

    problems: list[str] = []
    if wrong_dtype:
        name, found = wrong_dtype[0]
        problems.append(
            f"{len(wrong_dtype)} DequantizeLinear nodes take a non-8-bit input "
            f"(first: {name}, {found}). TensorRT accepts only FP8, FP4, Int4, "
            "Int8 or UInt8 there."
        )
    if nonzero_zp:
        problems.append(
            f"{len(nonzero_zp)} nodes have a non-zero zero point (first: "
            f"{nonzero_zp[0]}). TensorRT supports only symmetric quantization."
        )
    return problems


def has_qdq_nodes(onnx_path: Path) -> bool:
    """True when the graph carries QuantizeLinear/DequantizeLinear nodes.

    That is what makes a graph INT8 under TensorRT 11: the quantisation is
    baked in as Q/DQ pairs, and the builder reads precision from them rather
    than from a flag. `quantize.py --mode static` produces exactly this.

    Raises:
        MissingLFSContentError: ``onnx_path`` is an unfetched LFS pointer.
    """
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is a hard dependency
        return False

    reject_lfs_pointer(onnx_path)
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    ops = {node.op_type for node in graph.node}
    return "QuantizeLinear" in ops and "DequantizeLinear" in ops


def build_engine(
    onnx_path: Path,
    engine_path: Path | None = None,
    *,
    precision: str = "fp16",
    max_batch_size: int = 8,
    workspace_gb: float = 4.0,
    calibration_batches: list[np.ndarray] | None = None,
    verify: bool = True,
) -> TensorRTExportResult:
    """Compile an ONNX model into a TensorRT engine.

    Args:
        onnx_path: Source ONNX model.
        engine_path: Where to write the ``.engine`` file.
        precision: ``"fp32"``, ``"fp16"`` or ``"int8"``. fp16 is almost always
            the right default on modern NVIDIA hardware: roughly 2x faster
            than fp32 with negligible accuracy loss, and no calibration data.
        max_batch_size: Largest batch the engine will accept. An optimisation
            profile is built covering 1 to this value.
        workspace_gb: Scratch memory TensorRT may use while *building*. More
            workspace lets it consider faster but memory-hungrier kernels; it
            does not affect memory use at inference time.
        calibration_batches: Needed for ``precision="int8"`` only when
            ``onnx_path`` is a plain fp32 graph. A QDQ graph already carries
            the scales that calibration would produce, so none is required.

    Raises:
        RuntimeError: TensorRT is unavailable, or the build failed.
        ValueError: INT8 was requested from a plain graph with no calibration
            data.
        UnsupportedPrecisionError: INT8 was requested from a plain graph on a
            TensorRT that has no INT8 builder flag.
    """
    available, reason = tensorrt_available()
    if not available:
        raise RuntimeError(f"TensorRT cannot be used on this machine: {reason}")

    import tensorrt as trt

    onnx_path = Path(onnx_path)
    reject_lfs_pointer(onnx_path)
    engine_path = (
        Path(engine_path) if engine_path else onnx_path.with_suffix(f".{precision}.engine")
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # A QDQ graph was calibrated when it was quantised, so asking for
    # calibration data again would be asking twice for the same thing.
    qdq = precision == "int8" and has_qdq_nodes(onnx_path)
    if precision == "int8" and not qdq and not calibration_batches:
        raise ValueError(
            "INT8 engines need calibration data. Pass calibration_batches with a few "
            "hundred real images, build from a QDQ graph produced by "
            "`python -m models.optimization.quantize --mode static`, or build an "
            "fp16 engine instead."
        )

    class _CapturingLogger(trt.ILogger):
        """Print as before, but keep the messages so a failure can quote them."""

        def __init__(self) -> None:
            trt.ILogger.__init__(self)
            self.messages: list[tuple[str, str]] = []

        def log(self, severity: object, msg: str) -> None:
            name = getattr(severity, "name", str(severity))
            self.messages.append((name, msg))
            if name in ("ERROR", "INTERNAL_ERROR", "WARNING"):
                print(f"[TensorRT {name}] {msg}", file=sys.stderr)

        def errors(self) -> list[str]:
            return [m for level, m in self.messages if level in ("ERROR", "INTERNAL_ERROR")]

    logger = _CapturingLogger()
    builder = trt.Builder(logger)

    # On TensorRT 11 the precision lives in the graph, so an fp16 engine is
    # built from an fp16 ONNX. Do that conversion here rather than making
    # every caller know about it. The ORIGINAL fp32 model is still what the
    # engine is verified against below - comparing an fp16 engine to an fp16
    # graph would hide exactly the error we want to measure.
    source_onnx = onnx_path
    if not hasattr(trt.BuilderFlag, "FP16") and precision == "fp16":
        source_onnx = convert_onnx_to_fp16(onnx_path)

    # Explicit batch mode. TensorRT 8.x and 9.x require the EXPLICIT_BATCH
    # creation flag; TensorRT 10 made explicit batch the only mode and REMOVED
    # the flag, so referencing it there raises AttributeError. Probing for the
    # attribute keeps one code path working across both.
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    flags = 0 if explicit_batch is None else 1 << int(explicit_batch)

    # TensorRT 11 removed the precision BuilderFlags (FP16, INT8) and moved to
    # STRONGLY_TYPED networks, where precision comes from the dtypes in the
    # ONNX graph rather than from a builder switch. Detect that era by the
    # absence of BuilderFlag.FP16 rather than by parsing a version string.
    strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    typed_network_era = not hasattr(trt.BuilderFlag, "FP16")
    # A QDQ graph is strongly typed by construction: the Q/DQ pairs say what
    # precision each tensor is, which is precisely what the flag means.
    if typed_network_era and strongly_typed is not None and (precision != "int8" or qdq):
        flags |= 1 << int(strongly_typed)

    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(source_onnx.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT could not parse the ONNX model:\n  " + "\n  ".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    notes: list[str] = []

    # builder.platform_has_fast_fp16 / _int8 were deprecated in TensorRT 8.6
    # and REMOVED in 10, where querying them raises AttributeError. They were
    # only ever advisory: setting the flag on a GPU without fast support is
    # harmless, because TensorRT falls back to a supported precision per layer.
    # So probe for the attribute and skip the note when it is gone.
    def _platform_supports(attr: str) -> bool | None:
        value = getattr(builder, attr, None)
        return None if value is None else bool(value)

    if precision == "int8" and qdq:
        problems = check_trt_qdq_graph(onnx_path)
        if problems:
            raise UnsupportedPrecisionError(
                f"{onnx_path.name} is a QDQ graph, but not one TensorRT will build:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
                + "\n\nRegenerate it with:\n"
                "    python -m models.optimization.quantize --onnx <fp32.onnx> "
                "--mode static --trt-compatible --calibration-dir <images>\n"
                "which writes <name>_int8_trt.onnx with both constraints satisfied."
            )

    if typed_network_era and precision == "int8" and not qdq:
        # Strongly-typed era: there is no flag to set, so precision comes from
        # the graph. This ONNX is plain fp32, so building it would produce an
        # fp32 engine labelled INT8 - the kind of quiet wrong answer this
        # project has been bitten by before. Refuse, and say what to do.
        raise UnsupportedPrecisionError(
            f"TensorRT {trt.__version__} uses strongly-typed networks: "
            "BuilderFlag.INT8 no longer exists, and precision is taken from "
            f"the ONNX graph. {onnx_path.name} carries no QuantizeLinear nodes, "
            "so there is nothing to build an INT8 engine from. Produce a QDQ "
            "graph first:\n"
            "    python -m models.optimization.quantize --onnx "
            f"{onnx_path.name} --mode static --calibration-dir <images>\n"
            "then build the engine from the resulting "
            "<name>_int8_static.onnx."
        )

    if typed_network_era:
        # Nothing to set. The network was created STRONGLY_TYPED and the graph
        # itself is fp16 (convert_onnx_to_fp16 ran above), so precision is
        # already decided. Touching trt.BuilderFlag.FP16 here would raise
        # AttributeError, because that is exactly the attribute whose absence
        # defines this era.
        if precision == "fp16":
            notes.append(
                "fp16 from a converted fp16 ONNX graph; TensorRT "
                f"{trt.__version__} has no FP16 builder flag"
            )
        elif precision == "int8":
            notes.append(
                "int8 from a QDQ graph; precision comes from the "
                "QuantizeLinear/DequantizeLinear nodes rather than a builder flag"
            )
    elif precision == "fp16":
        if _platform_supports("platform_has_fast_fp16") is False:
            notes.append("this GPU has no fast fp16 support, so the engine will fall back to fp32")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if _platform_supports("platform_has_fast_int8") is False:
            notes.append("this GPU has no fast INT8 support; expect little or no speed-up")
        config.set_flag(trt.BuilderFlag.INT8)
        if qdq:
            # The Q/DQ nodes already carry the scales calibration would compute,
            # and attaching a calibrator on top of them makes TensorRT ignore
            # one of the two. Let the graph win: it is the one that was checked
            # for accuracy in benchmarks/reports/quantization.json.
            notes.append("int8 scales taken from the QDQ graph, so no calibrator was attached")
        else:
            config.int8_calibrator = _EntropyCalibrator(
                calibration_batches or [], engine_path.with_suffix(".calib")
            )

    # An optimisation profile tells TensorRT the range of input shapes to
    # expect. Without one, a dynamic ONNX model cannot be built at all.
    input_tensor = network.get_input(0)
    shape = list(input_tensor.shape)
    spatial = [d if d > 0 else 224 for d in shape[1:]]

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(1, *spatial),
        # `opt` is the shape TensorRT tunes hardest for. Batch 1 is the right
        # choice for a latency-sensitive API; raise it for a throughput-
        # oriented batch service.
        opt=(1, *spatial),
        max=(max_batch_size, *spatial),
    )
    config.add_optimization_profile(profile)

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)

    if (
        serialized is None
        and qdq
        and strongly_typed is not None
        and flags & (1 << int(strongly_typed))
    ):
        # STRONGLY_TYPED makes TensorRT honour the graph's types exactly, with
        # no per-layer fallback - so a single layer with no int8 kernel fails
        # the whole build. The Q/DQ nodes already say what should be
        # quantized, so dropping the flag lets TensorRT choose per layer
        # without turning this into an fp32 engine wearing an int8 label.
        print(
            "  strongly-typed int8 build failed; retrying with per-layer type selection",
            file=sys.stderr,
        )
        retry_flags = flags & ~(1 << int(strongly_typed))
        retry_network = builder.create_network(retry_flags)
        retry_parser = trt.OnnxParser(retry_network, logger)
        if retry_parser.parse(source_onnx.read_bytes()):
            retry_input = retry_network.get_input(0)
            retry_profile = builder.create_optimization_profile()
            retry_profile.set_shape(
                retry_input.name,
                min=(1, *spatial),
                opt=(1, *spatial),
                max=(max_batch_size, *spatial),
            )
            retry_config = builder.create_builder_config()
            retry_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
            )
            retry_config.add_optimization_profile(retry_profile)
            serialized = builder.build_serialized_network(retry_network, retry_config)
            if serialized is not None:
                network = retry_network
                notes.append(
                    "built without STRONGLY_TYPED: the strongly-typed build failed, most "
                    "likely a layer with no int8 kernel and no fallback allowed. Precision "
                    "still comes from the graph's Q/DQ nodes"
                )

    build_seconds = time.perf_counter() - started

    if serialized is None:
        reported = logger.errors()
        detail = (
            "\n\nTensorRT reported:\n" + "\n".join(f"  {m}" for m in reported[-8:])
            if reported
            else "\n\nTensorRT logged no error, which usually means it ran out of "
            "workspace while timing kernels. Try a larger workspace_gb."
        )
        raise RuntimeError(f"TensorRT failed to build the engine.{detail}")

    engine_path.write_bytes(serialized)

    import torch

    result = TensorRTExportResult(
        name=onnx_path.stem,
        engine_path=str(engine_path),
        precision=precision,
        onnx_mb=source_onnx.stat().st_size / 1_048_576,
        engine_mb=engine_path.stat().st_size / 1_048_576,
        build_seconds=build_seconds,
        max_batch_size=max_batch_size,
        gpu_name=torch.cuda.get_device_name(0),
        tensorrt_version=trt.__version__,
        notes=notes,
    )

    if verify:
        try:
            result.max_abs_diff, result.verified = _verify_engine(
                onnx_path, engine_path, spatial, precision
            )
        except Exception as exc:
            result.notes.append(f"verification failed: {type(exc).__name__}: {exc}")

    # An engine is not portable. Recording the hardware and version it was
    # built for turns a baffling future load failure into an obvious one.
    meta_path = engine_path.with_suffix(".json")
    meta_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    return result


#: How far the engine may drift from the ONNX graph before the build is
#: suspect, as a FRACTION of the reference output's peak magnitude.
#:
#: Relative, not absolute. These are logits, and their scale is a property of
#: the model - this classifier's span about +-6.7 - so an absolute bound means
#: something different on every model and silently tightens as the outputs
#: grow. An earlier absolute version failed the fp32 and fp16 engines of a
#: perfectly good build while passing INT8, which is exactly backwards.
#:
#: The values, and why each is where it is:
#:
#: * ``fp32`` 0.1% - not bit-exact, because TensorRT defaults to TF32 for fp32
#:   matmuls on Ampere and later. TF32 keeps 10 mantissa bits against fp32's
#:   23, so ~1e-3 relative error is the expected cost of the faster kernel.
#:   Measured on an A100: 0.042%.
#: * ``fp16`` 1% - half precision carries about three decimal digits.
#:   Measured: 0.209%.
#: * ``int8`` 10% - TensorRT and ONNX Runtime round and fuse the same QDQ
#:   graph differently, so a tight bound fails every honest build. Measured:
#:   3.305%.
#:
#: Loose as these are, they still catch what this check is for: a mis-parsed
#: graph or a wrong optimisation profile produces garbage, not drift.
_VERIFY_TOLERANCE = {"fp32": 1e-3, "fp16": 1e-2, "int8": 1e-1}


def _verify_engine(
    onnx_path: Path,
    engine_path: Path,
    spatial: list[int],
    precision: str = "fp16",
) -> tuple[float, bool]:
    """Run the ONNX model and the engine on the same input and compare."""
    import onnxruntime as ort

    from api.services.model_service import TensorRTBackend

    sample = np.random.randn(1, *spatial).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = session.run(None, {session.get_inputs()[0].name: sample})[0]

    backend = TensorRTBackend(engine_path)
    actual = backend.infer(sample)[0]
    backend.close()

    reference = reference.astype(np.float64)
    max_diff = float(np.abs(reference - actual.astype(np.float64)).max())

    # Scale by the reference's own magnitude so the bound means the same thing
    # on any model. The guard against a degenerate all-zero reference keeps a
    # broken engine from dividing its way to a pass.
    peak = float(np.abs(reference).max())
    relative = max_diff / peak if peak > 0 else float("inf")
    return max_diff, relative < _VERIFY_TOLERANCE.get(precision, 1e-2)


def benchmark_engine(
    engine_path: Path, *, input_shape: tuple[int, ...], iterations: int = 100, warmup: int = 20
) -> dict[str, float]:
    """Measure engine latency.

    Warmup matters far more here than with ONNX Runtime: the first several
    inferences on a GPU include kernel autotuning and memory allocation and
    can be an order of magnitude slower than the steady state.
    """
    from api.services.model_service import TensorRTBackend
    from models.optimization.benchmark import measure, summarise

    backend = TensorRTBackend(Path(engine_path))
    data = np.random.randn(*input_shape).astype(np.float32)

    timings = measure(lambda: backend.infer(data), iterations=iterations, warmup=warmup)
    result = summarise(
        timings,
        name=Path(engine_path).stem,
        runtime="tensorrt",
        device="cuda",
        batch_size=input_shape[0],
        size_mb=Path(engine_path).stat().st_size / 1_048_576,
    )
    backend.close()

    return {
        "p50_ms": round(result.p50_ms, 3),
        "p95_ms": round(result.p95_ms, 3),
        "p99_ms": round(result.p99_ms, 3),
        "throughput_ips": round(result.throughput_ips, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help=(
            "Real images. Needed for --precision int8 only when --onnx is a plain "
            "fp32 graph; a QDQ graph already carries its scales."
        ),
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "benchmarks" / "reports" / "tensorrt.json"
    )
    args = parser.parse_args()

    available, reason = tensorrt_available()
    if not available:
        print(f"TensorRT is not available on this machine: {reason}", file=sys.stderr)
        print(
            "\nThis is expected on a CPU-only host. The export code is written and gated;\n"
            "run it on a machine with an NVIDIA GPU after:\n"
            "  pip install -r requirements-gpu.txt",
            file=sys.stderr,
        )
        # Exit 0: "no GPU here" is a fact about the machine, not a failure of
        # this script, and failing CI over it would be wrong.
        return 0

    calibration_batches = None
    if args.precision == "int8" and not has_qdq_nodes(args.onnx):
        if not args.calibration_dir:
            print(
                f"error: {args.onnx.name} is a plain fp32 graph, so an INT8 engine needs\n"
                "       either --calibration-dir, or a QDQ graph built with:\n"
                "         python -m models.optimization.quantize --mode static",
                file=sys.stderr,
            )
            return 2
        from api.utils.image_processing import PreprocessConfig
        from models.optimization.quantize import iter_calibration_images

        cfg = PreprocessConfig(size=(args.image_size, args.image_size))
        calibration_batches = list(iter_calibration_images(args.calibration_dir, cfg, 200))
        print(f"loaded {len(calibration_batches)} calibration images")

    result = build_engine(
        args.onnx,
        args.output,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        workspace_gb=args.workspace_gb,
        calibration_batches=calibration_batches,
    )
    print(result.summary())
    for note in result.notes:
        print(f"  note: {note}")

    payload: dict[str, Any] = asdict(result)

    if args.benchmark:
        payload["benchmark"] = benchmark_engine(
            Path(result.engine_path), input_shape=(1, 3, args.image_size, args.image_size)
        )
        print(f"  benchmark: {payload['benchmark']}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report: {args.report}")

    print(
        "\nRegister the engine with:\n"
        f"  python -m models.registry register --name {result.name} --version 1.1.0 "
        f"--task classification --tensorrt {Path(result.engine_path).name} --overwrite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
