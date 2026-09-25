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
    """Stands in for a pycuda DeviceAllocation.

    It grew a `free()` because the real one has it and the backend now calls
    it. A fake missing a method the production code relies on is a fake that
    passes while production aborts, which is exactly what happened: device
    buffers were left to the garbage collector and the process died at exit
    on a real GPU.
    """

    def __new__(cls, value: int, freed: list[int] | None = None):
        ptr = super().__new__(cls, value)
        ptr._freed = freed if freed is not None else []
        return ptr

    def free(self) -> None:
        self._freed.append(int(self))


def _fake_trt_modules(monkeypatch, *, engine=None, deserialise_to_none: bool = False):
    """Install a minimal fake `tensorrt` + `pycuda` into sys.modules.

    Just enough surface for the backend: an engine with named IO tensors, an
    execution context that records what it was handed, and a cuda module whose
    allocations are counted so a leak shows up as a failed assertion.
    """
    copied: dict[int, np.ndarray] = {}
    allocated: list[int] = []
    freed: list[int] = []

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
        ptr = _FakeDevicePtr(len(allocated) + 1, freed)
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

    # A context that refuses CUDA work unless it has been pushed. The old
    # fake let every call through, which is why a backend that only worked on
    # one thread passed the whole suite and failed 6% of real requests.
    class _FakeContext:
        def __init__(self) -> None:
            self.depth = 0
            self.max_depth = 0
            self.pushes = 0
            self.detached = False

        def push(self) -> None:
            self.depth += 1
            self.pushes += 1
            self.max_depth = max(self.max_depth, self.depth)

        def pop(self) -> None:
            if self.depth == 0:
                raise RuntimeError("pop without a matching push")
            self.depth -= 1

        def detach(self) -> None:
            self.detached = True

    context = _FakeContext()

    def _require_current(what: str):
        if context.depth == 0:
            raise RuntimeError(f"{what}: invalid device context - no currently active context?")

    cuda.init = lambda: None
    cuda.Device = lambda index: SimpleNamespace(retain_primary_context=lambda: context)

    _real_stream = cuda.Stream

    def _guarded_stream():
        _require_current("Stream")
        return _real_stream()

    cuda.Stream = _guarded_stream

    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, "pycuda", SimpleNamespace(driver=cuda))
    monkeypatch.setitem(sys.modules, "pycuda.driver", cuda)
    monkeypatch.setitem(sys.modules, "pycuda.autoinit", SimpleNamespace())
    return SimpleNamespace(copied=copied, allocated=allocated, freed=freed, context=context)


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

    def test_inference_works_off_the_loading_thread(self, engine_file: Path, monkeypatch) -> None:
        """The bug that produced HTTP 500s on a real GPU.

        A CUDA context is thread-local, and inference runs through
        asyncio.to_thread, so it executes on whatever pool worker is free.
        With the context bound to the importing thread, every call on another
        worker failed with "invalid device context". Serving 100 requests
        sequentially on an A100 produced 6 of them.
        """
        import threading

        _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)

        results: list[object] = []

        def run() -> None:
            try:
                results.append(
                    backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
                )
            except Exception as exc:
                results.append(exc)

        # Several threads, none of them the one that loaded the engine.
        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        failures = [r for r in results if isinstance(r, Exception)]
        assert not failures, f"{len(failures)} of 4 threads failed: {failures[0]}"

    def test_the_context_is_pushed_and_popped_evenly(self, engine_file: Path, monkeypatch) -> None:
        """An unbalanced push leaks the context onto the thread, and PyCUDA
        aborts at shutdown with 'the context stack was not empty'."""
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        for _ in range(3):
            backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))

        assert fake.context.pushes > 0, "the context was never made current"
        assert (
            fake.context.depth == 0
        ), f"context stack left {fake.context.depth} deep; it must be balanced"

    def test_the_context_is_popped_even_when_inference_fails(
        self, engine_file: Path, monkeypatch
    ) -> None:
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)

        def explode(**kwargs):
            raise RuntimeError("execution failed")

        backend.context.execute_async_v3 = explode
        with pytest.raises(RuntimeError, match="execution failed"):
            backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))

        assert fake.context.depth == 0

    def test_the_engine_is_destroyed_under_a_live_context(
        self, engine_file: Path, monkeypatch
    ) -> None:
        """TensorRT destructors talk to CUDA.

        With no context current they acquire one themselves and leave it on
        the stack, and PyCUDA aborts at interpreter shutdown with "the context
        stack was not empty upon module cleanup". Seen on a real A100 after
        the request path was already correct.
        """
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)

        depth_when_destroyed: list[int] = []

        class _Watched:
            """Records the context depth at the moment it is collected."""

            def __del__(self) -> None:
                depth_when_destroyed.append(fake.context.depth)

        backend.engine = _Watched()
        backend.context = None
        backend.close()

        assert depth_when_destroyed, "the engine was never released"
        assert depth_when_destroyed[0] > 0, "the engine was destroyed with no CUDA context current"

    def test_close_leaves_the_context_stack_empty(self, engine_file: Path, monkeypatch) -> None:
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        backend.close()
        assert fake.context.depth == 0

    def test_close_releases_the_primary_context(self, engine_file: Path, monkeypatch) -> None:
        """Detach, not destroy: it is retained, and something else in the
        process may still hold it."""
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        backend.close()
        assert fake.context.detached is True

    def test_every_device_buffer_is_freed(self, engine_file: Path, monkeypatch) -> None:
        """Left to the garbage collector, these are released at an
        unpredictable time: the allocator fragments under load, and at process
        exit the buffers outlive the CUDA context and PyCUDA aborts."""
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))

        assert fake.allocated, "the test fake recorded no allocations at all"
        assert len(fake.freed) == len(fake.allocated), (
            f"allocated {len(fake.allocated)} device buffers and freed " f"{len(fake.freed)}"
        )

    def test_buffers_are_freed_even_when_inference_fails(
        self, engine_file: Path, monkeypatch
    ) -> None:
        """A failing engine must not leak the buffers of that call. Repeated
        failures would otherwise exhaust device memory."""
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)

        def explode(**kwargs):
            raise RuntimeError("execution failed")

        backend.context.execute_async_v3 = explode

        with pytest.raises(RuntimeError, match="execution failed"):
            backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))

        assert len(fake.freed) == len(fake.allocated)

    def test_repeated_inference_does_not_accumulate_buffers(
        self, engine_file: Path, monkeypatch
    ) -> None:
        fake = _fake_trt_modules(monkeypatch)
        backend = TensorRTBackend(engine_file)
        for _ in range(5):
            backend.infer(np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        assert len(fake.freed) == len(fake.allocated)


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


# ---------------------------------------------------------------------------
# Artifact fingerprinting
# ---------------------------------------------------------------------------
class TestArtifactFingerprint:
    """The cache key carries this so replaced weights invalidate their entries.

    Bumping the registry version on a weight change is the right discipline,
    but it is a manual step. When it is missed, the cache goes on serving
    predictions produced by a file that is no longer on disk - no error, no
    slowdown, just confidently wrong answers until the TTL expires. This makes
    the miss harmless rather than silent.
    """

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.write_bytes(content)

    def test_same_content_gives_the_same_fingerprint(
        self, service: ModelService, tmp_path: Path
    ) -> None:
        self._write(tmp_path / "m.onnx", b"weights-A")
        entry = _entry(artifacts={"onnx": "m.onnx"})
        assert service.artifact_fingerprint(entry) == service.artifact_fingerprint(entry)

    def test_different_content_gives_a_different_fingerprint(
        self, service: ModelService, tmp_path: Path
    ) -> None:
        """The whole point: new weights, new key."""
        path = tmp_path / "m.onnx"
        entry = _entry(artifacts={"onnx": "m.onnx"})

        self._write(path, b"weights-A")
        first = service.artifact_fingerprint(entry)
        self._write(path, b"weights-B-which-is-longer")
        second = service.artifact_fingerprint(entry)

        assert first != second

    def test_identical_content_recopied_keeps_the_same_fingerprint(
        self, service: ModelService, tmp_path: Path
    ) -> None:
        """Content, not mtime.

        A deploy that re-copies an unchanged artifact must not throw the cache
        away - that would mean a cold cache after every restart.
        """
        import os
        import time

        path = tmp_path / "m.onnx"
        entry = _entry(artifacts={"onnx": "m.onnx"})

        self._write(path, b"identical-bytes")
        first = service.artifact_fingerprint(entry)

        time.sleep(0.01)
        os.utime(path, None)  # touch: same bytes, new mtime
        assert service.artifact_fingerprint(entry) == first

    def test_a_missing_artifact_gives_a_stable_placeholder(self, service: ModelService) -> None:
        """A missing file is the loader's problem to report, not the key's."""
        entry = _entry(artifacts={"onnx": "not-there.onnx"})
        assert service.artifact_fingerprint(entry) == "absent"

    def test_an_entry_with_no_artifacts_at_all(self, service: ModelService) -> None:
        assert service.artifact_fingerprint(_entry(artifacts={})) == "absent"

    def test_it_is_short_enough_for_a_cache_key(
        self, service: ModelService, tmp_path: Path
    ) -> None:
        self._write(tmp_path / "m.onnx", b"x" * 4096)
        fingerprint = service.artifact_fingerprint(_entry(artifacts={"onnx": "m.onnx"}))
        assert len(fingerprint) == 12
        assert fingerprint.isalnum()

    def test_the_file_is_read_once_per_state(
        self, service: ModelService, tmp_path: Path, monkeypatch
    ) -> None:
        """Hashing a 95 MB artifact on every request would be absurd."""
        path = tmp_path / "m.onnx"
        self._write(path, b"y" * (2 << 20))
        entry = _entry(artifacts={"onnx": "m.onnx"})

        reads = {"n": 0}
        real_open = Path.open

        def counting_open(self, *args, **kwargs):
            if self.name == "m.onnx" and "b" in str(args[0] if args else kwargs.get("mode", "")):
                reads["n"] += 1
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", counting_open)
        for _ in range(5):
            service.artifact_fingerprint(entry)
        assert reads["n"] == 1, f"artifact was read {reads['n']} times"


class TestCacheKeysCarryTheFingerprint:
    def test_replacing_weights_changes_the_key(self, tmp_path: Path) -> None:
        """An end-to-end statement of the bug this prevents."""
        from api.services.cache_service import build_cache_key

        before = build_cache_key("classify", "imagehash", "m:1.0.0@aaaaaaaaaaaa", "onnx", {})
        after = build_cache_key("classify", "imagehash", "m:1.0.0@bbbbbbbbbbbb", "onnx", {})
        assert before != after

    def test_the_same_weights_reuse_the_key(self) -> None:
        from api.services.cache_service import build_cache_key

        a = build_cache_key("classify", "imagehash", "m:1.0.0@aaaaaaaaaaaa", "onnx", {})
        b = build_cache_key("classify", "imagehash", "m:1.0.0@aaaaaaaaaaaa", "onnx", {})
        assert a == b
