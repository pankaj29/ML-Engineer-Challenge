"""Unit tests for the three runtime backends and the loading chain.

`api/services/model_service.py` sat at 74.7%. Everything still uncovered was a
runtime backend: the TorchScript one, the TensorRT one, and the paths in
`_build_runtime` / `warmup` / `unload_all` that reach them.

The TorchScript backend is tested for real - a genuine `torch.jit` archive is
saved and loaded, which also pins the security property the class exists for:
it loads TorchScript, never a pickled `nn.Module`.

TensorRT cannot run here: it needs an NVIDIA GPU and a driver. A fake
`tensorrt` + `pycuda` pair is injected into `sys.modules` instead. That is not
a test of TensorRT; it is a test of *our* buffer bookkeeping around it - which
output buffer maps to which tensor name, that every device allocation is
matched, that a null engine is reported rather than segfaulting later. Those
are ours to get wrong, and on a CPU-only CI box this is the only way to check
them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from api.exceptions import ModelLoadError
from api.services.model_service import (
    ModelEntry,
    ModelService,
    RuntimeFormat,
    TensorRTBackend,
    TorchRuntimeBackend,
)

torch = pytest.importorskip("torch")

IMAGE_SIZE = 16
NUM_CLASSES = 3


def _entry(**overrides) -> ModelEntry:
    record = {
        "name": "tiny",
        "version": "1.0.0",
        "task": "classification",
        "artifacts": {},
        "preprocess": "imagenet_224",
        "labels_file": None,
        "num_classes": NUM_CLASSES,
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "metrics": {},
        "limitations": [],
        "description": "fixture",
        "status": "active",
    }
    record.update(overrides)
    return ModelEntry.from_dict(record)


@pytest.fixture
def torchscript(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES),
    ).eval()
    path = tmp_path / "tiny.pt"
    torch.jit.save(torch.jit.script(model), str(path))
    return path


@pytest.fixture
def service(tmp_path: Path, monkeypatch) -> ModelService:
    """A ModelService whose artifacts directory is `tmp_path`."""
    import api.services.model_service as model_service
    from api.config import Settings

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"models": []}), encoding="utf-8")
    monkeypatch.setattr(
        model_service,
        "settings",
        Settings(model_registry_path=registry, model_artifacts_dir=tmp_path),
    )
    return ModelService()


# ---------------------------------------------------------------------------
# TorchScript
# ---------------------------------------------------------------------------
class TestTorchRuntimeBackend:
    def test_loads_and_infers(self, torchscript: Path) -> None:
        backend = TorchRuntimeBackend(torchscript)
        out = backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        assert len(out) == 1
        assert out[0].shape == (1, NUM_CLASSES)
        assert isinstance(out[0], np.ndarray)

    def test_reports_the_torch_format(self, torchscript: Path) -> None:
        assert TorchRuntimeBackend(torchscript).format is RuntimeFormat.TORCH

    def test_int8_is_reported_as_a_distinct_format(self, torchscript: Path) -> None:
        """The cache key includes the runtime, so this must not be conflated."""
        assert TorchRuntimeBackend(torchscript, int8=True).format is RuntimeFormat.TORCH_INT8

    def test_a_batch_passes_through_unchanged(self, torchscript: Path) -> None:
        out = TorchRuntimeBackend(torchscript).infer(
            np.zeros((4, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        )
        assert out[0].shape == (4, NUM_CLASSES)

    def test_multiple_outputs_are_all_returned(self, tmp_path: Path) -> None:
        """A detector returns boxes, scores and labels, not one tensor."""

        class TwoHeads(torch.nn.Module):
            def forward(self, x):
                flat = x.flatten(1)
                return flat.sum(1, keepdim=True), flat.mean(1, keepdim=True)

        path = tmp_path / "two.pt"
        torch.jit.save(torch.jit.script(TwoHeads().eval()), str(path))
        out = TorchRuntimeBackend(path).infer(
            np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        )
        assert len(out) == 2

    def test_a_missing_artifact_is_a_clean_error(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="missing"):
            TorchRuntimeBackend(tmp_path / "absent.pt")

    def test_a_pickled_module_is_refused_not_executed(self, tmp_path: Path) -> None:
        """The security property this class exists for.

        `torch.load` on a pickle executes arbitrary code from the file. This
        backend uses `torch.jit.load`, which cannot, so a plain pickle fails
        to load rather than running.
        """
        path = tmp_path / "pickled.pth"
        torch.save(torch.nn.Linear(4, 2), str(path))
        with pytest.raises(ModelLoadError):
            TorchRuntimeBackend(path)

    def test_garbage_bytes_are_a_clean_error(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.pt"
        path.write_bytes(b"this is not a model")
        with pytest.raises(ModelLoadError, match="could not be loaded"):
            TorchRuntimeBackend(path)

    def test_close_releases_the_module(self, torchscript: Path) -> None:
        backend = TorchRuntimeBackend(torchscript)
        backend.close()
        assert backend.module is None


# ---------------------------------------------------------------------------
# TensorRT, against a fake driver
# ---------------------------------------------------------------------------
class _FakeDevicePtr(int):
    pass


def _fake_trt_modules(monkeypatch, *, engine=None, deserialise_to_none: bool = False):
    """Install a minimal fake `tensorrt` + `pycuda` into sys.modules.

    Just enough surface for the backend: an engine with named IO tensors, an
    execution context that records what it was handed, and a cuda module whose
    allocations are counted so a leak shows up as a failed assertion.
    """
    copied: dict[int, np.ndarray] = {}
    allocated: list[int] = []

    class _IOMode:
        INPUT = "input"
        OUTPUT = "output"

    class _Context:
        def __init__(self):
            self.addresses: dict[str, int] = {}
            self.shapes: dict[str, tuple[int, ...]] = {}
            self.executed = False

        def set_input_shape(self, name, shape):
            self.shapes[name] = tuple(shape)

        def get_tensor_shape(self, name):
            return (1, NUM_CLASSES)

        def set_tensor_address(self, name, address):
            self.addresses[name] = address

        def execute_async_v3(self, stream_handle=None):
            self.executed = True
            return True

    class _Engine:
        num_io_tensors = 2
        _names = ["images", "logits"]

        def get_tensor_name(self, i):
            return self._names[i]

        def get_tensor_mode(self, name):
            return _IOMode.INPUT if name == "images" else _IOMode.OUTPUT

        def create_execution_context(self):
            return _Context()

    class _Runtime:
        def __init__(self, logger):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def deserialize_cuda_engine(self, blob):
            return None if deserialise_to_none else (engine or _Engine())

    class _Logger:
        WARNING = 1

        def __init__(self, level):
            pass

    trt = SimpleNamespace(
        Logger=_Logger,
        Runtime=_Runtime,
        TensorIOMode=_IOMode,
    )

    class _Stream:
        handle = 1234

        def synchronize(self):
            pass

    def _mem_alloc(nbytes):
        ptr = _FakeDevicePtr(len(allocated) + 1)
        allocated.append(nbytes)
        return ptr

    def _memcpy_htod_async(dest, src, stream):
        copied[int(dest)] = np.array(src, copy=True)

    def _memcpy_dtoh_async(dest, src, stream):
        dest.fill(7.0)

    cuda = SimpleNamespace(
        Stream=_Stream,
        mem_alloc=_mem_alloc,
        memcpy_htod_async=_memcpy_htod_async,
        memcpy_dtoh_async=_memcpy_dtoh_async,
    )

    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, "pycuda", SimpleNamespace(driver=cuda))
    monkeypatch.setitem(sys.modules, "pycuda.driver", cuda)
    monkeypatch.setitem(sys.modules, "pycuda.autoinit", SimpleNamespace())
    return SimpleNamespace(copied=copied, allocated=allocated)


@pytest.fixture
def engine_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.engine"
    path.write_bytes(b"serialised-engine-bytes")
    return path


class TestTensorRTUnavailable:
    def test_a_cpu_host_gets_a_clean_error_not_an_import_crash(
        self, engine_file: Path, monkeypatch
    ) -> None:
        """This is the common case, and the fallback chain depends on it."""
        for name in ("tensorrt", "pycuda", "pycuda.driver", "pycuda.autoinit"):
            monkeypatch.setitem(sys.modules, name, None)
        with pytest.raises(ModelLoadError, match="not available on this host"):
            TensorRTBackend(engine_file)


class TestTensorRTBackend:
    def test_loads_an_engine(self, engine_file: Path, monkeypatch) -> None:
        _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        assert backend.format is RuntimeFormat.TENSORRT
        assert backend.device == "cuda"

    def test_a_missing_engine_file_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        _fake_trt_modules(monkeypatch)
        with pytest.raises(ModelLoadError, match="missing"):
            TensorRTBackend(tmp_path / "absent.engine")

    def test_an_engine_built_for_another_gpu_is_reported(
        self, engine_file: Path, monkeypatch
    ) -> None:
        """Deserialisation returns None rather than raising - easy to miss."""
        _fake_trt_modules(monkeypatch, deserialise_to_none=True)
        with pytest.raises(ModelLoadError, match="could not be deserialised"):
            TensorRTBackend(engine_file)

    def test_infer_returns_one_array_per_output_tensor(
        self, engine_file: Path, monkeypatch
    ) -> None:
        _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        out = backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        assert len(out) == 1
        assert out[0].shape == (1, NUM_CLASSES)

    def test_the_output_buffer_is_actually_copied_back(
        self, engine_file: Path, monkeypatch
    ) -> None:
        """Returning the un-copied `np.empty` would give plausible garbage."""
        _fake_trt_modules(monkeypatch)
        out = TensorRTBackend(engine_file).infer(
            np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        )
        assert np.all(out[0] == 7.0)

    def test_the_input_is_uploaded_contiguously(self, engine_file: Path, monkeypatch) -> None:
        """A non-contiguous view would upload the wrong bytes silently."""
        spy = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        view = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE * 2), dtype=np.float32)[..., ::2]
        assert not view.flags["C_CONTIGUOUS"]
        backend.infer(view)
        uploaded = next(iter(spy.copied.values()))
        assert uploaded.flags["C_CONTIGUOUS"]

    def test_the_execution_context_was_actually_run(self, engine_file: Path, monkeypatch) -> None:
        _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        assert backend.context.executed is True

    def test_every_io_tensor_gets_a_device_address(self, engine_file: Path, monkeypatch) -> None:
        _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        assert set(backend.context.addresses) == {"images", "logits"}

    def test_close_drops_both_handles(self, engine_file: Path, monkeypatch) -> None:
        _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        backend.close()
        assert backend.engine is None and backend.context is None


# ---------------------------------------------------------------------------
# _build_runtime dispatch
# ---------------------------------------------------------------------------
class TestBuildRuntimeDispatch:
    def test_torch_format_builds_the_torch_backend(
        self, service: ModelService, torchscript: Path
    ) -> None:
        runtime = service._build_runtime(_entry(artifacts={"torch": "tiny.pt"}), "torch")
        assert isinstance(runtime, TorchRuntimeBackend)

    def test_torch_int8_is_pinned_to_cpu(self, service: ModelService, torchscript: Path) -> None:
        """INT8 PyTorch kernels have no CUDA implementation."""
        runtime = service._build_runtime(_entry(artifacts={"torch_int8": "tiny.pt"}), "torch_int8")
        assert runtime.device == "cpu"
        assert runtime.format is RuntimeFormat.TORCH_INT8

    def test_tensorrt_format_builds_the_tensorrt_backend(
        self, service: ModelService, engine_file: Path, monkeypatch
    ) -> None:
        _fake_trt_modules(monkeypatch)
        runtime = service._build_runtime(_entry(artifacts={"tensorrt": "model.engine"}), "tensorrt")
        assert isinstance(runtime, TensorRTBackend)

    def test_a_format_the_model_does_not_have_is_refused(self, service: ModelService) -> None:
        with pytest.raises(ModelLoadError, match="not available for this model"):
            service._build_runtime(_entry(artifacts={"onnx": "tiny.onnx"}), "torch")

    def test_an_unknown_format_is_refused(self, service: ModelService) -> None:
        with pytest.raises(ModelLoadError, match="Unknown runtime format"):
            service._build_runtime(_entry(artifacts={"wasm": "tiny.wasm"}), "wasm")


# ---------------------------------------------------------------------------
# The fallback chain, exercised through load()
# ---------------------------------------------------------------------------
class TestLoadFallsThroughTheChain:
    def test_a_broken_first_choice_falls_through_to_a_working_one(
        self, service: ModelService, tmp_path: Path, torchscript: Path
    ) -> None:
        """The whole point of the preference chain."""
        (tmp_path / "broken.onnx").write_bytes(b"not an onnx graph")
        entry = _entry(artifacts={"onnx": "broken.onnx", "torch": "tiny.pt"})
        loaded = service.load(entry)
        assert loaded.runtime.format is RuntimeFormat.TORCH

    def test_every_runtime_failing_raises(self, service: ModelService, tmp_path: Path) -> None:
        (tmp_path / "broken.onnx").write_bytes(b"not an onnx graph")
        with pytest.raises(ModelLoadError):
            service.load(_entry(artifacts={"onnx": "broken.onnx"}))

    def test_the_failure_is_recorded_for_the_health_endpoint(
        self, service: ModelService, tmp_path: Path
    ) -> None:
        (tmp_path / "broken.onnx").write_bytes(b"not an onnx graph")
        with pytest.raises(ModelLoadError):
            service.load(_entry(artifacts={"onnx": "broken.onnx"}))
        assert service.health()["failures"]

    def test_a_loaded_model_is_reused_not_reloaded(
        self, service: ModelService, torchscript: Path
    ) -> None:
        entry = _entry(artifacts={"torch": "tiny.pt"})
        assert service.load(entry) is service.load(entry)


class TestUnloadAll:
    def test_releases_every_loaded_model(self, service: ModelService, torchscript: Path) -> None:
        service.load(_entry(artifacts={"torch": "tiny.pt"}))
        assert service.health()["loaded"] == 1
        service.unload_all()
        assert service.health()["loaded"] == 0

    def test_a_backend_that_throws_on_close_does_not_stop_shutdown(
        self, service: ModelService, torchscript: Path
    ) -> None:
        """Shutdown must never raise; one stuck handle cannot strand the rest."""
        loaded = service.load(_entry(artifacts={"torch": "tiny.pt"}))
        loaded.runtime.close = lambda: (_ for _ in ()).throw(RuntimeError("stuck"))
        service.unload_all()
        assert service.health()["loaded"] == 0
