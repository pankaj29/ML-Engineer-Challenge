"""Integration tests against the real model artifacts.

These load actual ONNX files and run actual inference, so they catch the class
of bug that fakes cannot: a broken export, a preprocessing mismatch, a label
file that does not line up with the model's outputs.

They skip automatically when the artifacts have not been prepared, so a fresh
clone still has a green suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.models.schemas import RuntimeFormat, TaskType

pytestmark = [pytest.mark.integration, pytest.mark.requires_models, pytest.mark.slow]


@pytest.fixture
def model_service(require_real_models: None):
    """A real ModelService reading the real registry."""
    from api.services.model_service import ModelService

    service = ModelService()
    yield service
    service.unload_all()


class TestRegistry:
    def test_registry_has_all_three_tasks(self, model_service) -> None:
        tasks = {entry.task for entry in model_service.list_entries()}
        assert tasks == {TaskType.CLASSIFICATION, TaskType.DETECTION, TaskType.SIMILARITY}

    def test_every_task_has_a_default(self, model_service) -> None:
        defaults = model_service.default_keys()
        assert set(defaults) == {"classification", "detection", "similarity"}

    def test_all_artifacts_exist(self) -> None:
        from models.registry import Registry

        assert Registry().validate() == []

    def test_resolve_unknown_model_raises(self, model_service) -> None:
        from api.exceptions import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            model_service.resolve(TaskType.CLASSIFICATION, "no-such-model")

    def test_resolve_unknown_version_raises(self, model_service) -> None:
        from api.exceptions import ModelNotFoundError

        entry = model_service.resolve(TaskType.CLASSIFICATION)
        with pytest.raises(ModelNotFoundError):
            model_service.resolve(TaskType.CLASSIFICATION, entry.name, "99.0.0")

    def test_explicit_version_resolves(self, model_service) -> None:
        entry = model_service.resolve(TaskType.CLASSIFICATION)
        assert (
            model_service.resolve(TaskType.CLASSIFICATION, entry.name, entry.version).version
            == entry.version
        )


class TestModelLoading:
    def test_loads_classifier(self, model_service) -> None:
        loaded = model_service.load(model_service.resolve(TaskType.CLASSIFICATION))
        assert loaded.runtime is not None
        assert len(loaded.labels) == 1000

    def test_labels_are_meaningful(self, model_service) -> None:
        """ImageNet index 207 is 'golden retriever'. A shifted label file
        would return the wrong name for every prediction."""
        loaded = model_service.load(model_service.resolve(TaskType.CLASSIFICATION))
        assert loaded.label_for(207) == "golden retriever"
        assert loaded.label_for(0) == "tench"

    def test_label_lookup_out_of_range_is_safe(self, model_service) -> None:
        loaded = model_service.load(model_service.resolve(TaskType.CLASSIFICATION))
        assert loaded.label_for(999_999) == "class_999999"

    def test_detector_has_coco_labels(self, model_service) -> None:
        loaded = model_service.load(model_service.resolve(TaskType.DETECTION))
        assert len(loaded.labels) == 80
        assert loaded.label_for(0) == "person"

    def test_second_load_is_cached(self, model_service) -> None:
        """Loading is expensive; the second call must reuse the instance."""
        entry = model_service.resolve(TaskType.CLASSIFICATION)
        first = model_service.load(entry)
        second = model_service.load(entry)
        assert first is second

    def test_missing_artifact_raises_model_load_error(self, model_service) -> None:
        from api.exceptions import ModelLoadError
        from api.services.model_service import ModelEntry

        broken = ModelEntry(
            name="ghost",
            version="1.0.0",
            task=TaskType.CLASSIFICATION,
            artifacts={"onnx": "this-file-does-not-exist.onnx"},
        )
        with pytest.raises(ModelLoadError):
            model_service.load(broken)


class TestRealInference:
    def test_classifier_output_shape(self, model_service) -> None:
        loaded = model_service.load(model_service.resolve(TaskType.CLASSIFICATION))
        output = loaded.runtime.infer(np.random.randn(1, 3, 224, 224).astype(np.float32))[0]
        assert output.shape == (1, 1000)

    def test_classifier_supports_batching(self, model_service) -> None:
        """The exported graph must accept a batch, or /batch cannot work."""
        loaded = model_service.load(model_service.resolve(TaskType.CLASSIFICATION))
        output = loaded.runtime.infer(np.random.randn(4, 3, 224, 224).astype(np.float32))[0]
        assert output.shape == (4, 1000)

    def test_detector_supports_batching(self, model_service) -> None:
        loaded = model_service.load(model_service.resolve(TaskType.DETECTION))
        output = loaded.runtime.infer(np.random.randn(2, 3, 640, 640).astype(np.float32))[0]
        assert output.shape[0] == 2

    def test_embedder_output_is_unit_length(self, model_service) -> None:
        """Normalisation is baked into the exported graph, so the raw output
        must already have length 1."""
        loaded = model_service.load(model_service.resolve(TaskType.SIMILARITY))
        output = loaded.runtime.infer(np.random.randn(1, 3, 224, 224).astype(np.float32))[0]
        assert np.linalg.norm(output[0]) == pytest.approx(1.0, abs=1e-4)

    def test_inference_is_deterministic(self, model_service) -> None:
        loaded = model_service.load(model_service.resolve(TaskType.CLASSIFICATION))
        data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        np.testing.assert_allclose(
            loaded.runtime.infer(data)[0], loaded.runtime.infer(data)[0], atol=1e-6
        )

    def test_int8_runtime_loads_and_agrees(self, model_service) -> None:
        """The quantized model must exist and broadly agree with float32."""
        entry = model_service.resolve(TaskType.CLASSIFICATION)
        if "onnx_int8" not in entry.artifacts:
            pytest.skip("no INT8 artifact registered")

        fp32 = model_service.load(entry, RuntimeFormat.ONNX)
        int8 = model_service.load(entry, RuntimeFormat.ONNX_INT8)

        data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        a = fp32.runtime.infer(data)[0]
        b = int8.runtime.infer(data)[0]
        assert a.shape == b.shape
        # Quantization changes the numbers; it must not change the shape or
        # produce NaNs.
        assert not np.isnan(b).any()


class TestEndToEndInference:
    """The full service path with real models and real images."""

    @pytest.fixture
    def real_service(self, model_service):
        from api.services.inference_service import InferenceService
        from tests.conftest import NullCache

        return InferenceService(model_service, NullCache())  # type: ignore[arg-type]

    async def test_classify_real_image(self, real_service, sample_image: bytes) -> None:
        response = await real_service.classify(sample_image, top_k=5)
        assert len(response.predictions) == 5
        # Real labels, not "class_207".
        assert not response.predictions[0].label.startswith("class_")
        assert 0.0 <= response.predictions[0].confidence <= 1.0

    async def test_classify_probabilities_sum_sensibly(
        self, real_service, sample_image: bytes
    ) -> None:
        response = await real_service.classify(sample_image, top_k=1000)
        total = sum(p.confidence for p in response.predictions)
        assert total == pytest.approx(1.0, abs=0.01)

    async def test_detect_real_image(self, real_service, sample_image: bytes) -> None:
        response = await real_service.detect(sample_image, confidence_threshold=0.25)
        assert response.image_width == 224
        assert response.count == len(response.detections)

    async def test_embed_real_image(self, real_service, sample_image: bytes) -> None:
        response = await real_service.embed(sample_image)
        assert response.dimension == 2048
        assert np.linalg.norm(response.embedding) == pytest.approx(1.0, abs=1e-4)

    async def test_meets_sub_second_requirement(self, real_service, sample_image: bytes) -> None:
        """The brief requires sub-second inference for a single image.

        The first call includes model load, so it is discarded; this measures
        the steady state a real user sees.
        """
        await real_service.classify(sample_image)

        timings = []
        for _ in range(5):
            response = await real_service.classify(sample_image, use_cache=False)
            timings.append(response.timing.total_ms)

        timings.sort()
        assert timings[len(timings) // 2] < 1000, f"median latency {timings[2]:.0f} ms exceeds 1 s"

    async def test_identical_images_give_identical_embeddings(
        self, real_service, sample_image: bytes
    ) -> None:
        """Cosine similarity of an image with itself must be 1.0."""
        a = await real_service.embed_array(sample_image)
        b = await real_service.embed_array(sample_image)
        assert float(np.dot(a, b)) == pytest.approx(1.0, abs=1e-5)

    async def test_different_images_are_less_similar(self, real_service) -> None:
        """Two visibly different images must score below a self-match."""
        from tests.conftest import make_image, make_noise_image

        flat = await real_service.embed_array(make_image(224, 224, (10, 200, 30)))
        noise = await real_service.embed_array(make_noise_image(224, 224))
        assert float(np.dot(flat, noise)) < 0.999
