"""Unit tests for model registry parsing, version resolution and fallbacks.

These use a temporary registry file and dummy artifacts, so they test the
resolution logic without loading real weights.
"""

from __future__ import annotations

import json

import pytest

from api.config import Settings
from api.exceptions import ModelLoadError, ModelNotFoundError
from api.models.schemas import RuntimeFormat, TaskType
from api.services.model_service import ModelEntry, ModelService


@pytest.fixture
def registry_file(tmp_path):
    """Build a registry with several versions across tasks."""

    def _write(models: list[dict]) -> Settings:
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"models": models}), encoding="utf-8")
        return Settings(
            environment="test",
            model_registry_path=path,
            model_artifacts_dir=tmp_path / "artifacts",
        )

    return _write


def entry_dict(name: str, version: str, task: str, **overrides) -> dict:
    record = {
        "name": name,
        "version": version,
        "task": task,
        "artifacts": {"onnx": f"{name}-{version}.onnx"},
        "preprocess": "imagenet_224",
        "num_classes": 1000,
        "input_shape": [1, 3, 224, 224],
        "is_default": False,
        "status": "active",
    }
    record.update(overrides)
    return record


class TestRegistryParsing:
    def test_loads_entries(self, registry_file) -> None:
        service = ModelService(registry_file([entry_dict("a", "1.0.0", "classification")]))
        assert len(service.list_entries()) == 1

    def test_missing_registry_is_tolerated(self, tmp_path) -> None:
        """A fresh clone with no registry must not crash the service."""
        service = ModelService(
            Settings(environment="test", model_registry_path=tmp_path / "absent.json")
        )
        assert service.list_entries() == []

    def test_malformed_json_is_tolerated(self, tmp_path) -> None:
        path = tmp_path / "registry.json"
        path.write_text("{ this is not json", encoding="utf-8")
        service = ModelService(Settings(environment="test", model_registry_path=path))
        assert service.list_entries() == []

    def test_bad_entry_is_skipped_not_fatal(self, registry_file) -> None:
        """One broken entry must not hide every other model."""
        service = ModelService(
            registry_file(
                [
                    entry_dict("good", "1.0.0", "classification"),
                    {"name": "bad", "task": "not-a-real-task"},
                ]
            )
        )
        assert [e.name for e in service.list_entries()] == ["good"]

    def test_retired_entries_are_excluded(self, registry_file) -> None:
        service = ModelService(
            registry_file(
                [
                    entry_dict("live", "1.0.0", "classification"),
                    entry_dict("old", "0.9.0", "classification", status="retired"),
                ]
            )
        )
        assert [e.name for e in service.list_entries()] == ["live"]

    def test_accepts_bare_list_format(self, tmp_path) -> None:
        path = tmp_path / "registry.json"
        path.write_text(json.dumps([entry_dict("a", "1.0.0", "classification")]), encoding="utf-8")
        service = ModelService(Settings(environment="test", model_registry_path=path))
        assert len(service.list_entries()) == 1


class TestVersionResolution:
    @pytest.fixture
    def service(self, registry_file) -> ModelService:
        return ModelService(
            registry_file(
                [
                    entry_dict("alpha", "1.0.0", "classification"),
                    entry_dict("alpha", "1.2.0", "classification"),
                    entry_dict("alpha", "2.0.0", "classification"),
                    entry_dict("beta", "1.0.0", "classification", is_default=True),
                    entry_dict("det", "1.0.0", "detection"),
                ]
            )
        )

    def test_latest_picks_highest_version(self, service: ModelService) -> None:
        assert service.resolve(TaskType.CLASSIFICATION, "alpha", "latest").version == "2.0.0"

    def test_no_version_means_latest(self, service: ModelService) -> None:
        assert service.resolve(TaskType.CLASSIFICATION, "alpha").version == "2.0.0"

    def test_exact_version_is_honoured(self, service: ModelService) -> None:
        assert service.resolve(TaskType.CLASSIFICATION, "alpha", "1.2.0").version == "1.2.0"

    def test_no_name_uses_the_task_default(self, service: ModelService) -> None:
        """The default flag wins over a numerically higher version."""
        assert service.resolve(TaskType.CLASSIFICATION).name == "beta"

    def test_version_sorting_is_numeric_not_lexical(self, registry_file) -> None:
        """'10.0.0' must beat '9.0.0'; string comparison would get this wrong."""
        service = ModelService(
            registry_file(
                [
                    entry_dict("m", "9.0.0", "classification"),
                    entry_dict("m", "10.0.0", "classification"),
                ]
            )
        )
        assert service.resolve(TaskType.CLASSIFICATION, "m").version == "10.0.0"

    def test_unknown_name_raises(self, service: ModelService) -> None:
        with pytest.raises(ModelNotFoundError) as exc:
            service.resolve(TaskType.CLASSIFICATION, "ghost")
        assert "available" in exc.value.details

    def test_unknown_version_raises_and_lists_options(self, service: ModelService) -> None:
        with pytest.raises(ModelNotFoundError) as exc:
            service.resolve(TaskType.CLASSIFICATION, "alpha", "99.0.0")
        assert "1.2.0" in exc.value.details["available_versions"]

    def test_task_with_no_models_raises(self, service: ModelService) -> None:
        with pytest.raises(ModelNotFoundError):
            service.resolve(TaskType.SIMILARITY)

    def test_default_keys_covers_every_available_task(self, service: ModelService) -> None:
        defaults = service.default_keys()
        assert defaults["classification"] == "beta:1.0.0"
        assert defaults["detection"] == "det:1.0.0"
        assert "similarity" not in defaults


class TestRuntimePreference:
    def test_prefers_configured_runtime(self, registry_file) -> None:
        settings = registry_file(
            [entry_dict("m", "1.0.0", "classification", artifacts={"onnx": "a", "torch": "b"})]
        )
        settings.preferred_runtime = "torch"
        service = ModelService(settings)
        entry = service.resolve(TaskType.CLASSIFICATION)
        assert service._runtime_preference(entry, None)[0] == "torch"

    def test_explicit_request_wins(self, registry_file) -> None:
        service = ModelService(
            registry_file(
                [
                    entry_dict(
                        "m",
                        "1.0.0",
                        "classification",
                        artifacts={"onnx": "a", "onnx_int8": "b"},
                    )
                ]
            )
        )
        entry = service.resolve(TaskType.CLASSIFICATION)
        assert service._runtime_preference(entry, RuntimeFormat.ONNX_INT8)[0] == "onnx_int8"

    def test_only_registered_formats_are_offered(self, registry_file) -> None:
        """We must not try to load a TensorRT engine that was never built."""
        service = ModelService(
            registry_file([entry_dict("m", "1.0.0", "classification", artifacts={"onnx": "a"})])
        )
        entry = service.resolve(TaskType.CLASSIFICATION)
        assert service._runtime_preference(entry, None) == ["onnx"]

    def test_no_artifacts_raises_on_load(self, registry_file) -> None:
        service = ModelService(
            registry_file([entry_dict("m", "1.0.0", "classification", artifacts={})])
        )
        entry = ModelEntry(name="m", version="1.0.0", task=TaskType.CLASSIFICATION, artifacts={})
        with pytest.raises(ModelLoadError):
            service.load(entry)

    def test_missing_file_raises_model_load_error(self, registry_file) -> None:
        service = ModelService(registry_file([entry_dict("m", "1.0.0", "classification")]))
        with pytest.raises(ModelLoadError) as exc:
            service.load(service.resolve(TaskType.CLASSIFICATION))
        assert "tried" in exc.value.details


class TestModelEntry:
    def test_key_format(self) -> None:
        entry = ModelEntry(name="m", version="1.2.3", task=TaskType.CLASSIFICATION)
        assert entry.key == "m:1.2.3"

    @pytest.mark.parametrize(
        ("version", "expected"),
        [("1.2.3", (1, 2, 3)), ("2.0", (2, 0)), ("1.0.0-beta", (1, 0, 0)), ("v3", (3,))],
    )
    def test_version_tuple(self, version: str, expected: tuple) -> None:
        entry = ModelEntry(name="m", version=version, task=TaskType.CLASSIFICATION)
        assert entry.version_tuple() == expected

    def test_non_numeric_version_does_not_crash(self) -> None:
        entry = ModelEntry(name="m", version="latest", task=TaskType.CLASSIFICATION)
        assert entry.version_tuple() == (0,)

    def test_preprocess_config_lookup(self) -> None:
        entry = ModelEntry(
            name="m", version="1.0.0", task=TaskType.DETECTION, preprocess="yolo_640"
        )
        assert entry.preprocess_config.size == (640, 640)

    def test_unknown_preprocess_falls_back(self) -> None:
        entry = ModelEntry(
            name="m", version="1.0.0", task=TaskType.CLASSIFICATION, preprocess="nonexistent"
        )
        assert entry.preprocess_config.size == (224, 224)

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """A future registry field must not break an older deployment."""
        entry = ModelEntry.from_dict(
            {
                "name": "m",
                "version": "1.0.0",
                "task": "classification",
                "some_future_field": "ignored",
            }
        )
        assert entry.name == "m"


class TestLabels:
    def test_loads_list_format(self, tmp_path, registry_file) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "labels.json").write_text(json.dumps(["cat", "dog"]), encoding="utf-8")

        settings = registry_file(
            [entry_dict("m", "1.0.0", "classification", labels_file="labels.json")]
        )
        service = ModelService(settings)
        assert service._load_labels(service.resolve(TaskType.CLASSIFICATION)) == ["cat", "dog"]

    def test_loads_dict_format_in_numeric_order(self, tmp_path, registry_file) -> None:
        """A {"0": ..., "1": ...} mapping must sort numerically, not as strings."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "labels.json").write_text(
            json.dumps({"10": "eleventh", "2": "third", "0": "first"}), encoding="utf-8"
        )
        settings = registry_file(
            [entry_dict("m", "1.0.0", "classification", labels_file="labels.json")]
        )
        service = ModelService(settings)
        labels = service._load_labels(service.resolve(TaskType.CLASSIFICATION))
        assert labels[0] == "first"
        assert labels[-1] == "eleventh"

    def test_missing_labels_file_is_a_warning_not_an_error(self, registry_file) -> None:
        """A model without labels still works; it just returns numeric names."""
        settings = registry_file(
            [entry_dict("m", "1.0.0", "classification", labels_file="absent.json")]
        )
        service = ModelService(settings)
        assert service._load_labels(service.resolve(TaskType.CLASSIFICATION)) == []


class TestHealth:
    def test_health_snapshot(self, registry_file) -> None:
        service = ModelService(registry_file([entry_dict("m", "1.0.0", "classification")]))
        health = service.health()
        assert health["registered"] == 1
        assert health["loaded"] == 0
        assert health["device"] in ("cpu", "cuda")

    def test_reload_picks_up_changes(self, tmp_path) -> None:
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"models": [entry_dict("a", "1.0.0", "classification")]}), encoding="utf-8"
        )
        service = ModelService(
            Settings(environment="test", model_registry_path=path, model_artifacts_dir=tmp_path)
        )
        assert len(service.list_entries()) == 1

        path.write_text(
            json.dumps(
                {
                    "models": [
                        entry_dict("a", "1.0.0", "classification"),
                        entry_dict("b", "1.0.0", "detection"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        service.reload_registry()
        assert len(service.list_entries()) == 2


class TestTensorRTRuntimeSelection:
    """What the GPU overlay depends on, checked without a GPU.

    The overlay sets PREFERRED_RUNTIME=tensorrt and registers an engine. None
    of that helps if the runtime chain ignores the setting, and the failure is
    invisible: the pod serves correct predictions from ONNX on CPU and looks
    healthy doing it.
    """

    @staticmethod
    def _entry(artifacts: dict[str, str]):
        from api.services.model_service import ModelEntry

        return ModelEntry(
            name="m",
            version="v",
            task="classification",
            artifacts=artifacts,
            preprocess="imagenet_224",
            labels_file=None,
            num_classes=200,
        )

    def test_the_preference_setting_puts_tensorrt_first(self, monkeypatch) -> None:
        from api.services.model_service import ModelService

        service = ModelService()
        monkeypatch.setattr(service.settings, "preferred_runtime", "tensorrt")

        chain = service._runtime_preference(
            self._entry({"onnx": "m.onnx", "tensorrt": "m.engine"}), None
        )
        assert chain[0] == "tensorrt", (
            f"PREFERRED_RUNTIME=tensorrt but the chain starts with {chain[0]!r}, "
            "so the GPU overlay would serve from CPU and report success"
        )

    def test_an_engine_only_entry_has_nothing_to_fall_back_to(self) -> None:
        """How the GPU proof is registered. With ONNX alongside, a TensorRT
        failure serves ONNX and returns 200, which proves nothing."""
        from api.services.model_service import ModelService

        chain = ModelService()._runtime_preference(self._entry({"tensorrt": "m.engine"}), None)
        assert chain == ["tensorrt"]

    def test_asking_for_tensorrt_on_a_cpu_host_fails_cleanly(self, tmp_path) -> None:
        """Not with an ImportError from somewhere inside the backend."""
        import pytest

        from api.exceptions import ModelLoadError
        from api.services.model_service import TensorRTBackend

        pytest.importorskip  # noqa: B018 - referenced so the intent is clear
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            with pytest.raises(ModelLoadError, match="TensorRT is not available"):
                TensorRTBackend(tmp_path / "missing.engine")
        else:
            with pytest.raises(ModelLoadError, match="engine file is missing"):
                TensorRTBackend(tmp_path / "missing.engine")


class TestTensorRTTeardown:
    """Resource release, which only misbehaves on a machine with a GPU.

    Found by running the API on a real A100: every process that served a
    prediction aborted at exit with

        have been deinitialized, so there is no way we can finish cleanly.
        The program will be aborted now.

    `pycuda.autoinit` destroys the CUDA context from an atexit handler, and
    the engine was still alive when it ran. In Kubernetes that turns every
    SIGTERM into a non-zero exit, so a normal rolling update looks like a
    crashing container.
    """

    def test_close_is_registered_to_run_before_pycuda_tears_down(self) -> None:
        """atexit runs handlers last-registered-first.

        The registration has to come after the pycuda.autoinit import, or it
        runs second and the context is already gone.
        """
        import inspect

        from api.services.model_service import TensorRTBackend

        source = inspect.getsource(TensorRTBackend.__init__)
        assert (
            "atexit.register(self.close)" in source
        ), "nothing releases the engine before pycuda destroys the context"
        assert source.index("pycuda.autoinit") < source.index("atexit.register"), (
            "registered before the autoinit import, so it runs after pycuda's "
            "own handler and the context is already destroyed"
        )

    def test_device_buffers_are_freed_explicitly(self) -> None:
        """Relying on refcounting frees them at an unpredictable time, which
        fragments the allocator under load and outlives the context at exit."""
        import inspect

        from api.services.model_service import TensorRTBackend

        source = inspect.getsource(TensorRTBackend.infer)
        assert (
            "finally:" in source and ".free()" in source
        ), "device allocations are left to the garbage collector"

    def test_close_is_idempotent(self) -> None:
        """Both atexit and the model service's unload call it."""
        from api.services.model_service import TensorRTBackend

        backend = TensorRTBackend.__new__(TensorRTBackend)
        backend._closed = False
        backend.context = object()
        backend.engine = object()

        backend.close()
        assert backend.engine is None
        backend.close()  # must not raise

    def test_the_context_is_released_before_the_engine(self) -> None:
        """The execution context holds a reference to the engine."""
        import inspect

        from api.services.model_service import TensorRTBackend

        source = inspect.getsource(TensorRTBackend.close)
        assert source.index("self.context = None") < source.index("self.engine = None")
