"""Model registry, loading and version management.

Plain English:
    This is the service that answers "give me the model that can classify
    images" and hands back something with a ``.predict()`` method — without
    the caller ever needing to know whether that model is a PyTorch
    checkpoint, an ONNX graph or a TensorRT engine.

Three ideas do all the work here:

1. **A registry.** A JSON file lists every model we have: its name, versions,
   which task it performs, where each compiled artifact lives, its
   preprocessing config and its measured metrics. Adding a model means adding
   a registry entry, not changing code.

2. **A runtime abstraction.** :class:`ModelRuntime` is a small interface with
   one method, ``infer()``. ``OnnxRuntimeBackend`` and ``TorchRuntimeBackend``
   implement it. Everything upstream is written against the interface, so
   swapping ONNX for TensorRT changes one config value.

3. **Lazy, cached, thread-safe loading.** Models are expensive to load
   (hundreds of MB, seconds of time) and cheap to reuse. We load each one
   once, keep it in a dict, and guard that dict with a lock so that two
   simultaneous first-requests cannot load the same model twice.

Version resolution: a request may pin ``model_version="1.2.0"``, ask for
``"latest"``, or say nothing at all (the task default). All three paths funnel
through :meth:`ModelService.resolve`.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from api.config import Settings, settings
from api.exceptions import ModelLoadError, ModelNotFoundError
from api.logging_config import get_logger
from api.models.schemas import RuntimeFormat, TaskType
from api.utils.image_processing import (
    CLASSIFICATION_PREPROCESS,
    DETECTION_PREPROCESS,
    SIMILARITY_PREPROCESS,
    TINY_IMAGENET_PREPROCESS,
    PreprocessConfig,
)

logger = get_logger(__name__)

# Named preprocessing configs, referenced by string from the registry JSON so
# that the registry file stays pure data.
PREPROCESS_PRESETS: dict[str, PreprocessConfig] = {
    "imagenet_224": CLASSIFICATION_PREPROCESS,
    "tiny_imagenet": TINY_IMAGENET_PREPROCESS,
    "yolo_640": DETECTION_PREPROCESS,
    "clip_224": SIMILARITY_PREPROCESS,
}


# ---------------------------------------------------------------------------
# Registry data model
# ---------------------------------------------------------------------------
@dataclass
class ModelEntry:
    """One registered model version.

    Attributes:
        name: Registry name, e.g. ``"resnet50-tiny-imagenet"``.
        version: Semantic-ish version string, e.g. ``"1.0.0"``.
        task: Which task it performs.
        artifacts: Map of :class:`RuntimeFormat` value to artifact path,
            e.g. ``{"onnx": "resnet50.onnx", "torch": "resnet50.pt"}``.
            Paths are relative to ``settings.model_artifacts_dir``.
        preprocess: Name of a preset in :data:`PREPROCESS_PRESETS`.
        labels_file: JSON file with the ordered class-name list.
        metrics: Measured accuracy/latency numbers, surfaced by GET /models.
        limitations: Known failure modes, copied from the model card.
        is_default: Whether this serves requests that do not pin a version.
    """

    name: str
    version: str
    task: TaskType
    artifacts: dict[str, str] = field(default_factory=dict)
    preprocess: str = "imagenet_224"
    labels_file: str | None = None
    num_classes: int | None = None
    input_shape: list[int] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    description: str | None = None
    is_default: bool = False
    registered_at: str | None = None
    status: str = "active"

    @property
    def key(self) -> str:
        """Unique ``name:version`` identifier."""
        return f"{self.name}:{self.version}"

    @property
    def preprocess_config(self) -> PreprocessConfig:
        """The :class:`PreprocessConfig` this model needs."""
        return PREPROCESS_PRESETS.get(self.preprocess, CLASSIFICATION_PREPROCESS)

    def version_tuple(self) -> tuple[int, ...]:
        """Numeric version for sorting. Non-numeric parts sort as 0."""
        parts: list[int] = []
        for chunk in self.version.split("."):
            digits = "".join(c for c in chunk if c.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        """Build an entry from a registry JSON object, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in data.items() if k in known}
        payload["task"] = TaskType(payload["task"])
        return cls(**payload)


# ---------------------------------------------------------------------------
# Runtime abstraction
# ---------------------------------------------------------------------------
class ModelRuntime(Protocol):
    """Anything that can run a forward pass.

    Keeping this as a Protocol (structural typing) means a test can pass in a
    plain fake object with an ``infer`` method — no inheritance required.
    """

    format: RuntimeFormat
    device: str

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        """Run one forward pass and return the raw output arrays."""
        ...

    def close(self) -> None:
        """Release memory / GPU handles."""
        ...


class OnnxRuntimeBackend:
    """Runs a model through ONNX Runtime.

    This is the production default. ONNX Runtime applies graph optimisations
    (operator fusion, constant folding) that eager PyTorch does not, and it
    drops the Python interpreter out of the hot path entirely.
    """

    def __init__(self, path: Path, device: str = "cpu", *, int8: bool = False) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                "ONNX Runtime is not installed.", internal_message=str(exc)
            ) from exc

        if not path.exists():
            raise ModelLoadError(
                "The model artifact is missing.",
                details={"runtime": "onnx"},
                internal_message=f"artifact not found: {path}",
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Leave threading to the container's CPU limit rather than letting ORT
        # spawn a thread per core inside a cgroup-limited container.
        opts.intra_op_num_threads = 0

        try:
            self.session = ort.InferenceSession(str(path), opts, providers=providers)
        except Exception as exc:
            raise ModelLoadError(
                "The ONNX model could not be loaded.",
                internal_message=f"{type(exc).__name__}: {exc}",
            ) from exc

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.format = RuntimeFormat.ONNX_INT8 if int8 else RuntimeFormat.ONNX
        # Report what ORT actually selected, which may be CPU even if we asked
        # for CUDA (missing driver, no onnxruntime-gpu installed).
        self.device = "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"

        logger.info(
            "onnx_model_loaded",
            extra={"path": path.name, "device": self.device, "outputs": len(self.output_names)},
        )

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        return self.session.run(self.output_names, {self.input_name: inputs})

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]


class TorchRuntimeBackend:
    """Runs a TorchScript module. The fallback when no ONNX artifact exists.

    We deliberately load TorchScript (``torch.jit``) rather than a pickled
    ``nn.Module``. SECURITY: unpickling a ``.pth`` executes arbitrary code from
    the file, so a compromised artifact would be remote code execution.
    TorchScript archives carry no such risk.
    """

    def __init__(self, path: Path, device: str = "cpu", *, int8: bool = False) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError("PyTorch is not installed.", internal_message=str(exc)) from exc

        if not path.exists():
            raise ModelLoadError(
                "The model artifact is missing.",
                details={"runtime": "torch"},
                internal_message=f"artifact not found: {path}",
            )

        self._torch = torch
        try:
            self.module = torch.jit.load(str(path), map_location=device)
            self.module.eval()
        except Exception as exc:
            raise ModelLoadError(
                "The PyTorch model could not be loaded.",
                internal_message=f"{type(exc).__name__}: {exc}",
            ) from exc

        self.device = device
        self.format = RuntimeFormat.TORCH_INT8 if int8 else RuntimeFormat.TORCH
        logger.info("torch_model_loaded", extra={"path": path.name, "device": device})

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        torch = self._torch
        # inference_mode is faster than no_grad: it also skips version
        # counting on tensors, which we never need at serving time.
        with torch.inference_mode():
            tensor = torch.from_numpy(inputs).to(self.device)
            out = self.module(tensor)
        if isinstance(out, (list, tuple)):
            return [o.detach().cpu().numpy() for o in out]
        return [out.detach().cpu().numpy()]

    def close(self) -> None:
        self.module = None  # type: ignore[assignment]


class TensorRTBackend:
    """Runs a serialised TensorRT engine.

    TensorRT requires an NVIDIA GPU and the ``tensorrt`` package, neither of
    which exists on a CPU-only host. The import is therefore done inside
    ``__init__`` so that merely importing this module never fails; asking for
    a TensorRT runtime on a CPU box raises a clean
    :class:`~api.exceptions.ModelLoadError` that the fallback chain catches.
    """

    def __init__(self, path: Path, device: str = "cuda") -> None:
        try:
            import pycuda.autoinit  # noqa: F401  (side effect: creates CUDA context)
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise ModelLoadError(
                "TensorRT is not available on this host.",
                details={"runtime": "tensorrt"},
                internal_message=f"tensorrt/pycuda import failed: {exc}",
            ) from exc

        if not path.exists():
            raise ModelLoadError(
                "The TensorRT engine file is missing.",
                internal_message=f"engine not found: {path}",
            )

        self._cuda = cuda
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with path.open("rb") as fh, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise ModelLoadError(
                "The TensorRT engine could not be deserialised.",
                internal_message=(
                    "Engines are built for one specific GPU + TensorRT version "
                    "and are not portable across either."
                ),
            )
        self.context = self.engine.create_execution_context()
        self.format = RuntimeFormat.TENSORRT
        self.device = device
        self._tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        logger.info("tensorrt_engine_loaded", extra={"path": path.name})

    def infer(self, inputs: np.ndarray) -> list[np.ndarray]:
        cuda = self._cuda
        import tensorrt as trt

        inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        stream = cuda.Stream()
        allocations: list[Any] = []
        outputs: list[np.ndarray] = []

        for name in self._tensor_names:
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, inputs.shape)
                d_in = cuda.mem_alloc(inputs.nbytes)
                cuda.memcpy_htod_async(d_in, inputs, stream)
                self.context.set_tensor_address(name, int(d_in))
                allocations.append(d_in)
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                host = np.empty(shape, dtype=np.float32)
                d_out = cuda.mem_alloc(host.nbytes)
                self.context.set_tensor_address(name, int(d_out))
                allocations.append(d_out)
                outputs.append(host)

        self.context.execute_async_v3(stream_handle=stream.handle)

        out_idx = 0
        for name in self._tensor_names:
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                cuda.memcpy_dtoh_async(
                    outputs[out_idx], allocations[self._tensor_names.index(name)], stream
                )
                out_idx += 1
        stream.synchronize()
        return outputs

    def close(self) -> None:
        self.context = None
        self.engine = None


def _record_load(model: str, runtime: str, status: str) -> None:
    """Publish a model load attempt to Prometheus (best-effort)."""
    try:
        from api.middleware.monitoring import record_model_load

        record_model_load(model, runtime, status)
    except Exception:  # pragma: no cover
        pass


def _set_loaded_gauge(count: int) -> None:
    """Publish how many models are resident (best-effort)."""
    try:
        from api.middleware.monitoring import models_loaded

        models_loaded.set(count)
    except Exception:  # pragma: no cover
        pass


@dataclass
class LoadedModel:
    """A registry entry paired with its live runtime."""

    entry: ModelEntry
    runtime: ModelRuntime
    labels: list[str]
    loaded_at: float

    @property
    def preprocess_config(self) -> PreprocessConfig:
        return self.entry.preprocess_config

    def label_for(self, class_id: int) -> str:
        """Human-readable name for a class index, with a safe fallback."""
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
class ModelService:
    """Loads, caches and resolves models.

    One instance lives for the lifetime of the process, created during
    application startup and shared by every request.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or settings
        self.registry_path = Path(self.settings.model_registry_path)
        self.artifacts_dir = Path(self.settings.model_artifacts_dir)
        self.device = self.settings.resolve_device()

        self._entries: dict[str, ModelEntry] = {}
        self._loaded: dict[str, LoadedModel] = {}
        # Guards _loaded. A plain threading.Lock is right here because model
        # loading is synchronous, CPU-bound work that runs in a thread pool.
        self._lock = threading.RLock()
        self._load_failures: dict[str, str] = {}

        self.reload_registry()

    # ------------------------------------------------------------ registry --
    def reload_registry(self) -> None:
        """Re-read the registry JSON from disk.

        Called at startup, and callable again to pick up a newly registered
        model without restarting the service.
        """
        if not self.registry_path.exists():
            logger.warning(
                "registry_missing",
                extra={"path": str(self.registry_path)},
            )
            self._entries = {}
            return

        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("registry_parse_failed", extra={"error": str(exc)})
            self._entries = {}
            return

        # Accept both {"models": [...]} and a bare list, so a hand-written
        # registry does not have to get the wrapper exactly right.
        records = raw.get("models", raw) if isinstance(raw, dict) else raw
        if isinstance(records, dict):  # {"name:version": {...}} form
            records = list(records.values())

        entries: dict[str, ModelEntry] = {}
        for record in records:
            try:
                entry = ModelEntry.from_dict(record)
            except (KeyError, TypeError, ValueError) as exc:
                logger.error(
                    "registry_entry_invalid", extra={"error": str(exc), "record": str(record)[:200]}
                )
                continue
            if entry.status != "active":
                continue
            entries[entry.key] = entry

        self._entries = entries
        logger.info(
            "registry_loaded",
            extra={"count": len(entries), "models": sorted(entries)},
        )

    def list_entries(self) -> list[ModelEntry]:
        """Every active registry entry."""
        return sorted(
            self._entries.values(), key=lambda e: (e.task.value, e.name, e.version_tuple())
        )

    def is_loaded(self, key: str) -> bool:
        """True when ``name:version`` is resident in memory."""
        return key in self._loaded

    def load_failure(self, key: str) -> str | None:
        """Why a model failed to load, if it did."""
        return self._load_failures.get(key)

    # ----------------------------------------------------------- resolution --
    def resolve(
        self,
        task: TaskType,
        name: str | None = None,
        version: str | None = None,
    ) -> ModelEntry:
        """Pick the registry entry that should serve a request.

        Resolution order:
            1. If ``name`` and a concrete ``version`` are given, use exactly that.
            2. If ``name`` is given with ``"latest"`` (or nothing), use that
               model's highest version.
            3. If neither is given, use the task's default model, or its
               highest-versioned model if no default is flagged.

        Raises:
            ModelNotFoundError: Nothing in the registry satisfies the request.
        """
        version = version or "latest"
        candidates = [e for e in self._entries.values() if e.task == task]

        if not candidates:
            raise ModelNotFoundError(
                f"No model is registered for the '{task.value}' task.",
                details={
                    "task": task.value,
                    "available_tasks": sorted({e.task.value for e in self._entries.values()}),
                },
            )

        if name:
            candidates = [e for e in candidates if e.name == name]
            if not candidates:
                raise ModelNotFoundError(
                    f"No model named '{name}' is registered for the '{task.value}' task.",
                    details={
                        "requested": name,
                        "available": sorted(
                            {e.name for e in self._entries.values() if e.task == task}
                        ),
                    },
                )

        if version != "latest":
            exact = [e for e in candidates if e.version == version]
            if not exact:
                raise ModelNotFoundError(
                    f"Version '{version}' of that model is not registered.",
                    details={
                        "requested_version": version,
                        "available_versions": sorted(e.version for e in candidates),
                    },
                )
            return exact[0]

        # "latest": prefer the flagged default, otherwise the highest version.
        if not name:
            defaults = [e for e in candidates if e.is_default]
            if defaults:
                return max(defaults, key=lambda e: e.version_tuple())
        return max(candidates, key=lambda e: e.version_tuple())

    def default_keys(self) -> dict[str, str]:
        """The ``name:version`` chosen for each task when nothing is pinned."""
        out: dict[str, str] = {}
        for task in TaskType:
            try:
                out[task.value] = self.resolve(task).key
            except ModelNotFoundError:
                continue
        return out

    # -------------------------------------------------------------- loading --
    def _artifact_path(self, entry: ModelEntry, fmt: str) -> Path | None:
        rel = entry.artifacts.get(fmt)
        if not rel:
            return None
        path = Path(rel)
        return path if path.is_absolute() else self.artifacts_dir / path

    def _runtime_preference(self, entry: ModelEntry, requested: RuntimeFormat | None) -> list[str]:
        """Ordered list of runtime formats to try for this model.

        The caller's explicit choice comes first; after that we fall back
        through progressively more available options. This ordering is what
        makes graceful degradation work: if the quantised ONNX artifact is
        corrupt, we quietly serve from full-precision ONNX instead of failing.
        """
        preferred = requested.value if requested else self.settings.preferred_runtime
        chain = [
            preferred,
            "onnx",
            "onnx_int8",
            "torch",
            "torch_int8",
            "tensorrt",
        ]
        seen: set[str] = set()
        ordered = [f for f in chain if f in entry.artifacts and not (f in seen or seen.add(f))]
        return ordered

    def _build_runtime(self, entry: ModelEntry, fmt: str) -> ModelRuntime:
        path = self._artifact_path(entry, fmt)
        if path is None:
            raise ModelLoadError(
                "The requested runtime format is not available for this model.",
                details={"runtime": fmt, "available": sorted(entry.artifacts)},
            )
        if fmt in ("onnx", "onnx_int8"):
            return OnnxRuntimeBackend(path, self.device, int8=fmt.endswith("int8"))
        if fmt in ("torch", "torch_int8"):
            # INT8 quantised PyTorch models only execute on CPU.
            device = "cpu" if fmt.endswith("int8") else self.device
            return TorchRuntimeBackend(path, device, int8=fmt.endswith("int8"))
        if fmt == "tensorrt":
            return TensorRTBackend(path, "cuda")
        raise ModelLoadError(
            "Unknown runtime format.", internal_message=f"unsupported format {fmt!r}"
        )

    def _load_labels(self, entry: ModelEntry) -> list[str]:
        """Read the class-name list for a model.

        A missing labels file is a warning, not an error: the API still works,
        it just returns ``class_17`` instead of ``goldfish``.
        """
        if not entry.labels_file:
            return []
        path = Path(entry.labels_file)
        if not path.is_absolute():
            path = self.artifacts_dir / path
        if not path.exists():
            logger.warning("labels_missing", extra={"model": entry.key, "path": str(path)})
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("labels_parse_failed", extra={"model": entry.key, "error": str(exc)})
            return []
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            # {"0": "tench", "1": "goldfish", ...} — sort by numeric key.
            return [str(v) for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
        return []

    def load(self, entry: ModelEntry, requested: RuntimeFormat | None = None) -> LoadedModel:
        """Load a model, reusing the cached instance when possible.

        Tries each runtime in the preference chain and returns the first that
        loads successfully, so a single broken artifact cannot take the
        endpoint down.

        Raises:
            ModelLoadError: Every runtime format failed.
        """
        cache_key = f"{entry.key}:{requested.value if requested else 'auto'}"

        with self._lock:
            cached = self._loaded.get(cache_key)
            if cached is not None:
                return cached

            formats = self._runtime_preference(entry, requested)
            if not formats:
                raise ModelLoadError(
                    "This model has no usable artifacts registered.",
                    details={"model": entry.key},
                    internal_message=f"entry.artifacts is empty for {entry.key}",
                )

            errors: list[str] = []
            for fmt in formats:
                started = time.perf_counter()
                try:
                    runtime = self._build_runtime(entry, fmt)
                except ModelLoadError as exc:
                    errors.append(f"{fmt}: {exc.internal_message or exc.message}")
                    logger.warning(
                        "model_runtime_load_failed",
                        extra={"model": entry.key, "runtime": fmt, "reason": exc.internal_message},
                    )
                    _record_load(entry.key, fmt, "failed")
                    continue

                loaded = LoadedModel(
                    entry=entry,
                    runtime=runtime,
                    labels=self._load_labels(entry),
                    loaded_at=time.time(),
                )
                self._loaded[cache_key] = loaded
                self._load_failures.pop(entry.key, None)
                _record_load(entry.key, fmt, "success")
                _set_loaded_gauge(len(self._loaded))
                logger.info(
                    "model_loaded",
                    extra={
                        "model": entry.key,
                        "runtime": fmt,
                        "device": runtime.device,
                        "load_ms": round((time.perf_counter() - started) * 1000, 1),
                        "fell_back": fmt != formats[0],
                    },
                )
                return loaded

            reason = "; ".join(errors)
            self._load_failures[entry.key] = reason
            raise ModelLoadError(
                "The model could not be loaded in any available format.",
                details={"model": entry.key, "tried": formats},
                internal_message=reason,
            )

    async def load_async(
        self, entry: ModelEntry, requested: RuntimeFormat | None = None
    ) -> LoadedModel:
        """Async wrapper around :meth:`load`.

        Loading is blocking, CPU-bound work. Running it in a worker thread
        keeps the event loop free to serve other requests while a large model
        is being read off disk.
        """
        return await asyncio.to_thread(self.load, entry, requested)

    async def get(
        self,
        task: TaskType,
        name: str | None = None,
        version: str | None = None,
        runtime: RuntimeFormat | None = None,
    ) -> LoadedModel:
        """Resolve and load in one call — the method routers actually use."""
        entry = self.resolve(task, name, version)
        return await self.load_async(entry, runtime)

    async def warmup(self) -> dict[str, str]:
        """Preload the default model for every task at application startup.

        Why bother: the first inference through ONNX Runtime allocates arenas
        and JITs kernels, which can take several seconds. Doing that during
        startup means the first real user does not pay for it. Failures are
        recorded, not raised — a missing detector should not stop the
        classifier from serving.

        Returns:
            ``{"task": "loaded" | "failed: reason"}`` for each task.
        """
        report: dict[str, str] = {}
        for task in TaskType:
            try:
                entry = self.resolve(task)
            except ModelNotFoundError as exc:
                report[task.value] = f"skipped: {exc.message}"
                continue
            try:
                loaded = await self.load_async(entry)
            except ModelLoadError as exc:
                report[task.value] = f"failed: {exc.internal_message or exc.message}"
                logger.error("warmup_failed", extra={"task": task.value, "model": entry.key})
                continue

            # A real forward pass with zeros, so the very first user request
            # hits an already-warm graph.
            try:
                cfg = loaded.preprocess_config
                shape = (1, 3, cfg.size[0], cfg.size[1])
                await asyncio.to_thread(loaded.runtime.infer, np.zeros(shape, dtype=np.float32))
                report[task.value] = f"loaded: {entry.key}"
            except Exception as exc:  # a warmup failure is non-fatal
                report[task.value] = f"loaded (warmup pass failed): {entry.key}"
                logger.warning(
                    "warmup_inference_failed",
                    extra={"model": entry.key, "error": f"{type(exc).__name__}: {exc}"},
                )
        return report

    def unload_all(self) -> None:
        """Release every loaded model. Called during application shutdown."""
        with self._lock:
            for key, loaded in list(self._loaded.items()):
                try:
                    loaded.runtime.close()
                except Exception:  # shutdown must never raise
                    logger.debug("runtime_close_failed", extra={"model": key})
            self._loaded.clear()
        logger.info("models_unloaded")

    def health(self) -> dict[str, Any]:
        """Snapshot of loading state, used by the health endpoint."""
        return {
            "registered": len(self._entries),
            "loaded": len(self._loaded),
            "loaded_keys": sorted(self._loaded),
            "failures": dict(self._load_failures),
            "device": self.device,
        }


# Process-wide singleton, created in the application lifespan handler.
_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """FastAPI dependency returning the shared :class:`ModelService`."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def set_model_service(service: ModelService | None) -> None:
    """Replace the singleton. Used by the lifespan handler and by tests."""
    global _model_service
    _model_service = service


__all__ = [
    "PREPROCESS_PRESETS",
    "LoadedModel",
    "ModelEntry",
    "ModelRuntime",
    "ModelService",
    "OnnxRuntimeBackend",
    "TensorRTBackend",
    "TorchRuntimeBackend",
    "get_model_service",
    "set_model_service",
]
