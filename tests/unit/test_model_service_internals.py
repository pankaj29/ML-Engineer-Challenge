"""Unit tests for model loading, the runtime backends and warmup.

`api/services/model_service.py` was the largest uncovered module in `api/`
(68%), and it is the component that decides which model answers a request and
what happens when one will not load. The fallback chain in particular is
load-bearing: it is what turns "the GPU engine is broken" into "served from
ONNX on the CPU" rather than an outage.

The backends are exercised through their **failure** paths, which is where the
interesting behaviour lives and which needs no GPU, no TorchScript archive and
no 100 MB artifact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from api.exceptions import ModelLoadError, ModelNotFoundError
from api.models.schemas import RuntimeFormat, TaskType
from api.services.model_service import (
    LoadedModel,
    ModelEntry,
    ModelService,
    OnnxRuntimeBackend,
    TensorRTBackend,
    TorchRuntimeBackend,
)


@pytest.fixture
def make_service(tmp_path):
    """A real ModelService pointed at an empty temp registry.

    Constructed through __init__ rather than __new__, so `settings`,
    `artifacts_dir` and the device resolution are all genuinely exercised.
    """
    import json

    from api.config import Settings

    def _make(entries: list[ModelEntry] | None = None) -> ModelService:
        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps({"models": []}), encoding="utf-8")
        cfg = Settings(
            model_registry_path=str(registry),
            model_artifacts_dir=str(tmp_path),
        )
        service = ModelService(config=cfg)
        if entries:
            service._entries = {e.key: e for e in entries}
        return service

    return _make


def _entry(**overrides) -> ModelEntry:
    base = {
        "name": "m",
        "version": "1.0.0",
        "task": TaskType.CLASSIFICATION,
        "artifacts": {"onnx": "m.onnx"},
        "preprocess": "classification",
        "labels_file": None,
        "num_classes": 10,
        "input_shape": [1, 3, 224, 224],
        "is_default": True,
        "status": "active",
    }
    base.update(overrides)
    return ModelEntry.from_dict(base)


class TestModelEntry:
    def test_key_is_name_and_version(self) -> None:
        assert _entry().key == "m:1.0.0"

    def test_versions_sort_numerically_not_lexically(self) -> None:
        """'1.10.0' must be newer than '1.9.0'. String sorting says otherwise."""
        assert _entry(version="1.10.0").version_tuple() > _entry(version="1.9.0").version_tuple()

    def test_preprocess_config_resolves_from_the_preset_name(self) -> None:
        assert _entry(preprocess="classification").preprocess_config.size == (224, 224)

    def test_unknown_preset_falls_back_rather_than_crashing(self) -> None:
        assert _entry(preprocess="no-such-preset").preprocess_config is not None


class TestLoadedModel:
    def test_label_lookup(self) -> None:
        loaded = LoadedModel(entry=_entry(), runtime=None, labels=["cat", "dog"], loaded_at=0.0)
        assert loaded.label_for(1) == "dog"

    def test_out_of_range_label_degrades_to_a_placeholder(self) -> None:
        """Never raise on a label lookup - the prediction is still useful."""
        loaded = LoadedModel(entry=_entry(), runtime=None, labels=["cat"], loaded_at=0.0)
        assert loaded.label_for(999) == "class_999"

    def test_no_labels_at_all_still_returns_something(self) -> None:
        loaded = LoadedModel(entry=_entry(), runtime=None, labels=[], loaded_at=0.0)
        assert loaded.label_for(3) == "class_3"


class TestBackendFailures:
    """Every backend must fail with ModelLoadError, never a raw exception.

    The fallback chain catches ModelLoadError specifically. A backend that
    raised anything else would abort the chain instead of letting the next
    runtime try.
    """

    def test_onnx_backend_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError):
            OnnxRuntimeBackend(tmp_path / "absent.onnx")

    def test_onnx_backend_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.onnx"
        path.write_bytes(b"this is not a protobuf")
        with pytest.raises(ModelLoadError):
            OnnxRuntimeBackend(path)

    def test_torch_backend_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError):
            TorchRuntimeBackend(tmp_path / "absent.pt")

    def test_torch_backend_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pt"
        path.write_bytes(b"not a torchscript archive")
        with pytest.raises(ModelLoadError):
            TorchRuntimeBackend(path)

    def test_tensorrt_backend_fails_cleanly_without_a_gpu(self, tmp_path: Path) -> None:
        """On a CPU host this must be a catchable ModelLoadError, not ImportError."""
        path = tmp_path / "engine.plan"
        path.write_bytes(b"\x00" * 16)
        with pytest.raises(ModelLoadError):
            TensorRTBackend(path)


class TestOnnxBackendBehaviour:
    """Exercised against a tiny real ONNX model built on the fly."""

    @pytest.fixture(scope="class")
    def tiny_onnx(self, tmp_path_factory) -> Path:
        torch = pytest.importorskip("torch")
        path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 5)).eval()
        kwargs = {}
        if "dynamo" in torch.onnx.export.__code__.co_varnames:
            kwargs["dynamo"] = False
        torch.onnx.export(
            model,
            torch.randn(1, 3, 4, 4),
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            **kwargs,
        )
        return path

    def test_loads_and_infers(self, tiny_onnx: Path) -> None:
        backend = OnnxRuntimeBackend(tiny_onnx)
        out = backend.infer(np.zeros((1, 3, 4, 4), dtype=np.float32))
        assert out[0].shape == (1, 5)
        backend.close()

    def test_reports_the_format_it_is(self, tiny_onnx: Path) -> None:
        backend = OnnxRuntimeBackend(tiny_onnx)
        assert backend.format == RuntimeFormat.ONNX
        backend.close()

    def test_int8_flag_changes_the_reported_format(self, tiny_onnx: Path) -> None:
        """The runtime label matters: it is part of the cache key."""
        backend = OnnxRuntimeBackend(tiny_onnx, int8=True)
        assert backend.format == RuntimeFormat.ONNX_INT8
        backend.close()

    def test_supports_batching(self, tiny_onnx: Path) -> None:
        backend = OnnxRuntimeBackend(tiny_onnx)
        assert backend.infer(np.zeros((4, 3, 4, 4), dtype=np.float32))[0].shape == (4, 5)
        backend.close()


class TestRegistryResolution:
    """Version resolution, and the deliberate refusal to guess."""

    def test_resolves_the_task_default(self, make_service) -> None:
        service = make_service([_entry(is_default=True)])
        assert service.resolve(TaskType.CLASSIFICATION).key == "m:1.0.0"

    def test_resolves_an_exact_version_pin(self, make_service) -> None:
        service = make_service([_entry(version="1.0.0"), _entry(version="2.0.0")])
        assert service.resolve(TaskType.CLASSIFICATION, "m", "2.0.0").version == "2.0.0"

    def test_picks_the_highest_version_when_unpinned(self, make_service) -> None:
        service = make_service(
            [_entry(version="1.0.0"), _entry(version="1.10.0"), _entry(version="1.9.0")]
        )
        assert service.resolve(TaskType.CLASSIFICATION, "m").version == "1.10.0"

    def test_unknown_name_raises_rather_than_substituting(self, make_service) -> None:
        """Serving a different model than asked for is worse than an error."""
        service = make_service([_entry()])
        with pytest.raises(ModelNotFoundError):
            service.resolve(TaskType.CLASSIFICATION, "no-such-model")

    def test_unknown_version_raises(self, make_service) -> None:
        service = make_service([_entry(version="1.0.0")])
        with pytest.raises(ModelNotFoundError):
            service.resolve(TaskType.CLASSIFICATION, "m", "9.9.9")

    def test_no_model_for_the_task_raises(self, make_service) -> None:
        service = make_service([_entry()])
        with pytest.raises(ModelNotFoundError):
            service.resolve(TaskType.DETECTION)

    def test_list_entries_and_default_keys(self, make_service) -> None:
        service = make_service([_entry()])
        assert [e.key for e in service.list_entries()] == ["m:1.0.0"]
        assert service.default_keys()["classification"] == "m:1.0.0"


class TestRuntimePreference:
    """The fallback chain: order matters, and unavailable formats are skipped."""

    def test_only_offers_formats_the_entry_actually_has(self, make_service) -> None:
        service = make_service()
        entry = _entry(artifacts={"onnx": "m.onnx"})
        assert service._runtime_preference(entry, None) == ["onnx"]

    def test_requested_runtime_is_tried_first(self, make_service) -> None:
        service = make_service()
        entry = _entry(artifacts={"onnx": "m.onnx", "onnx_int8": "m8.onnx"})
        assert service._runtime_preference(entry, RuntimeFormat.ONNX_INT8)[0] == "onnx_int8"

    def test_order_is_deduplicated(self, make_service) -> None:
        service = make_service()
        entry = _entry(artifacts={"onnx": "m.onnx"})
        order = service._runtime_preference(entry, RuntimeFormat.ONNX)
        assert len(order) == len(set(order))

    def test_load_failure_lists_everything_it_tried(self, make_service) -> None:
        service = make_service()
        entry = _entry(artifacts={"onnx": "absent.onnx"})
        with pytest.raises(ModelLoadError):
            service.load(entry)


class TestWarmupAndHealth:
    async def test_warmup_reports_a_skip_when_no_model_exists(self, make_service) -> None:
        """A missing detector must not stop the classifier from serving."""
        report = await make_service().warmup()
        assert report
        assert all("skipped" in v or "failed" in v for v in report.values())

    async def test_warmup_covers_every_task(self, make_service) -> None:
        report = await make_service().warmup()
        assert set(report) == {t.value for t in TaskType}

    def test_health_reports_counts_without_loading_anything(self, make_service) -> None:
        health = make_service().health()
        assert health["registered"] == 0
        assert health["loaded"] == 0

    def test_unload_all_is_safe_when_nothing_is_loaded(self, make_service) -> None:
        service = make_service()
        service.unload_all()
        assert service.health()["loaded"] == 0

    async def test_load_async_surfaces_the_same_error_as_load(self, make_service) -> None:
        service = make_service()
        with pytest.raises(ModelLoadError):
            await asyncio.wait_for(
                service.load_async(_entry(artifacts={"onnx": "absent.onnx"})), timeout=10
            )
