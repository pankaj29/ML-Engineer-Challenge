"""Unit tests for graceful degradation in the inference service.

Part 2 of the brief requires "graceful degradation: fallback mechanisms when
models fail". That behaviour lived in `_resolve_model` and was untested, which
is a poor place to have no tests: the whole point of a fallback is that it runs
during an incident, when nobody is watching it closely.

The distinction these tests pin down is the important one - **fallback applies
to failures, not to typos**. A GPU engine that will not load should quietly
drop to ONNX on the CPU. A caller who asks for a model that does not exist must
get a 404, because silently answering with a *different* model is worse than an
error: they would act on confident predictions from something they never chose.
"""

from __future__ import annotations

import pytest

from api.exceptions import ModelLoadError, ModelNotFoundError
from api.models.schemas import TaskType
from api.services.inference_service import InferenceService


class _Models:
    """A model service whose behaviour each test dictates."""

    def __init__(self, *, default, on_request=None):
        self._default = default
        self._on_request = on_request
        self.calls: list[tuple] = []

    async def get(self, task, name=None, version=None, runtime=None):
        self.calls.append((task, name, version, runtime))
        if name is None and version is None and runtime is None:
            if isinstance(self._default, Exception):
                raise self._default
            return self._default
        if self._on_request is None:
            return self._default
        if isinstance(self._on_request, Exception):
            raise self._on_request
        return self._on_request


def _loaded(fake_model_service, name="test-classifier"):
    return fake_model_service.models["classification"]


class TestFallbackOnFailure:
    async def test_a_broken_pinned_model_falls_back_to_the_default(
        self, fake_model_service, null_cache
    ) -> None:
        default = _loaded(fake_model_service)
        models = _Models(default=default, on_request=ModelLoadError("engine will not load"))
        service = InferenceService(models, null_cache)

        model, degraded, warnings = await service._resolve_model(
            TaskType.CLASSIFICATION, "some-other-model", "9.9.9", None
        )
        assert model is default
        assert degraded is True
        assert warnings and "unavailable" in warnings[0]

    async def test_the_warning_names_both_models(self, fake_model_service, null_cache) -> None:
        """The caller must be able to tell what they actually got."""
        default = _loaded(fake_model_service)
        models = _Models(default=default, on_request=ModelLoadError("boom"))
        service = InferenceService(models, null_cache)

        _, _, warnings = await service._resolve_model(
            TaskType.CLASSIFICATION, "wanted", "1.2.3", None
        )
        assert "wanted" in warnings[0]
        assert default.entry.key in warnings[0]

    async def test_nothing_pinned_means_nothing_to_fall_back_to(self, null_cache) -> None:
        """A failing default has no alternative - raise rather than loop."""
        models = _Models(default=ModelLoadError("default is broken"))
        service = InferenceService(models, null_cache)

        with pytest.raises(ModelLoadError):
            await service._resolve_model(TaskType.CLASSIFICATION, None, None, None)

    async def test_no_fallback_when_the_default_is_the_thing_that_failed(
        self, fake_model_service, null_cache
    ) -> None:
        """Reporting a fallback that did not happen would be a lie."""
        default = _loaded(fake_model_service)
        models = _Models(default=default, on_request=ModelLoadError("broken"))
        service = InferenceService(models, null_cache)

        with pytest.raises(ModelLoadError):
            await service._resolve_model(
                TaskType.CLASSIFICATION,
                default.entry.name,
                default.entry.version,
                None,
            )

    async def test_the_original_error_survives_when_the_fallback_also_fails(
        self, null_cache
    ) -> None:
        """Report why the requested model failed, not why the backup did."""
        models = _Models(
            default=ModelLoadError("default also broken"),
            on_request=ModelLoadError("requested model broken"),
        )
        service = InferenceService(models, null_cache)

        with pytest.raises(ModelLoadError, match="requested model broken"):
            await service._resolve_model(TaskType.CLASSIFICATION, "x", "1.0.0", None)


class TestNoFallbackForUnknownModels:
    async def test_an_unknown_name_is_never_substituted(
        self, fake_model_service, null_cache
    ) -> None:
        """The single most important rule in this module."""
        models = _Models(
            default=_loaded(fake_model_service),
            on_request=ModelNotFoundError("no such model"),
        )
        service = InferenceService(models, null_cache)

        with pytest.raises(ModelNotFoundError):
            await service._resolve_model(TaskType.CLASSIFICATION, "typo-model", None, None)

    async def test_it_does_not_even_try_the_default(self, fake_model_service, null_cache) -> None:
        models = _Models(
            default=_loaded(fake_model_service),
            on_request=ModelNotFoundError("no such model"),
        )
        service = InferenceService(models, null_cache)

        with pytest.raises(ModelNotFoundError):
            await service._resolve_model(TaskType.CLASSIFICATION, "typo", None, None)
        assert len(models.calls) == 1, "a fallback was attempted for an unknown model"


class TestHappyPathIsNotDegraded:
    async def test_a_working_model_reports_no_degradation(
        self, fake_model_service, null_cache
    ) -> None:
        default = _loaded(fake_model_service)
        service = InferenceService(_Models(default=default), null_cache)

        model, degraded, warnings = await service._resolve_model(
            TaskType.CLASSIFICATION, None, None, None
        )
        assert model is default
        assert degraded is False
        assert warnings == []
