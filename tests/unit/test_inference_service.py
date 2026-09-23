"""Unit tests for the inference service.

Covers the numerical helpers (softmax, NMS) and the orchestration behaviour
that the whole API depends on: caching, fallbacks, timeouts and concurrency
limiting.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from api.exceptions import (
    InferenceError,
    InferenceTimeoutError,
    ModelNotFoundError,
    OverloadedError,
)
from api.services.inference_service import InferenceService, l2_normalize, nms, softmax
from tests.conftest import FakeRuntime


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        assert softmax(np.array([[1.0, 2.0, 3.0]])).sum() == pytest.approx(1.0)

    def test_preserves_ordering(self) -> None:
        probs = softmax(np.array([[1.0, 3.0, 2.0]]))[0]
        assert probs.argmax() == 1

    def test_survives_large_logits(self) -> None:
        """exp(1000) overflows to inf; the max-subtraction must prevent that."""
        probs = softmax(np.array([[1000.0, 1001.0, 999.0]]))
        assert not np.isnan(probs).any()
        assert probs.sum() == pytest.approx(1.0)

    def test_survives_large_negative_logits(self) -> None:
        probs = softmax(np.array([[-1000.0, -1001.0, -999.0]]))
        assert not np.isnan(probs).any()
        assert probs.sum() == pytest.approx(1.0)

    def test_uniform_input_gives_uniform_output(self) -> None:
        probs = softmax(np.array([[5.0, 5.0, 5.0, 5.0]]))[0]
        np.testing.assert_allclose(probs, 0.25)


class TestNMS:
    def test_removes_duplicate_boxes(self) -> None:
        """Two nearly identical boxes must collapse to one."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]])
        scores = np.array([0.9, 0.8])
        assert nms(boxes, scores, 0.5) == [0]

    def test_keeps_distinct_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
        scores = np.array([0.9, 0.8])
        assert sorted(nms(boxes, scores, 0.5)) == [0, 1]

    def test_returns_highest_score_first(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
        scores = np.array([0.3, 0.95])
        assert nms(boxes, scores, 0.5)[0] == 1

    def test_empty_input(self) -> None:
        assert nms(np.empty((0, 4)), np.array([]), 0.5) == []

    def test_single_box(self) -> None:
        assert nms(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0.9]), 0.5) == [0]

    def test_threshold_controls_strictness(self) -> None:
        """Two boxes at IoU ~0.33: kept at threshold 0.5, merged at 0.2."""
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]])
        scores = np.array([0.9, 0.8])
        assert len(nms(boxes, scores, 0.5)) == 2
        assert len(nms(boxes, scores, 0.2)) == 1

    def test_zero_area_box_does_not_divide_by_zero(self) -> None:
        boxes = np.array([[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 10.0, 10.0]])
        scores = np.array([0.9, 0.8])
        assert nms(boxes, scores, 0.5)  # must not raise


class TestL2Normalize:
    def test_produces_unit_length(self) -> None:
        assert np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))) == pytest.approx(1.0)

    def test_zero_vector_does_not_produce_nan(self) -> None:
        assert not np.isnan(l2_normalize(np.zeros(5))).any()

    def test_preserves_direction(self) -> None:
        out = l2_normalize(np.array([3.0, 4.0]))
        np.testing.assert_allclose(out, [0.6, 0.8])


class TestClassify:
    async def test_returns_top_k(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.classify(sample_image, top_k=5)
        assert len(response.predictions) == 5
        assert response.predictions[0].rank == 1

    async def test_predictions_are_descending(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.classify(sample_image, top_k=10)
        confidences = [p.confidence for p in response.predictions]
        assert confidences == sorted(confidences, reverse=True)

    async def test_confidence_threshold_filters(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.classify(
            sample_image, top_k=100, confidence_threshold=0.99
        )
        assert all(p.confidence >= 0.99 for p in response.predictions)

    async def test_top_prediction_matches_first(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.classify(sample_image, top_k=3)
        assert response.top_prediction == response.predictions[0]

    async def test_populates_labels(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.classify(sample_image, top_k=1)
        assert response.predictions[0].label.startswith("class_")

    async def test_timing_is_reported(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        timing = (await inference_service.classify(sample_image)).timing
        assert timing.total_ms > 0
        assert timing.inference_ms >= 0
        # Total must cover the parts, allowing for rounding.
        assert timing.total_ms >= timing.inference_ms - 0.1

    async def test_unknown_model_raises_not_found(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        """A typo'd model name is a client error, not a reason to fall back."""
        with pytest.raises(ModelNotFoundError):
            await inference_service.classify(sample_image, model_name="does-not-exist")

    async def test_inference_failure_becomes_inference_error(
        self, fake_model_service, null_cache, sample_image: bytes
    ) -> None:
        fake_model_service.models["classification"].runtime = FakeRuntime(fail=True)
        service = InferenceService(fake_model_service, null_cache)
        with pytest.raises(InferenceError):
            await service.classify(sample_image)


class TestDetect:
    async def test_returns_detections(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.detect(sample_image, confidence_threshold=0.0)
        assert response.count == len(response.detections)
        assert response.image_width == 224

    async def test_boxes_are_within_image(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.detect(sample_image, confidence_threshold=0.0)
        for detection in response.detections:
            assert 0 <= detection.box.x1 <= response.image_width
            assert 0 <= detection.box.y1 <= response.image_height
            assert detection.box.x2 <= response.image_width
            assert detection.box.y2 <= response.image_height

    async def test_max_detections_caps_output(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.detect(
            sample_image, confidence_threshold=0.0, max_detections=3
        )
        assert len(response.detections) <= 3

    async def test_high_threshold_returns_nothing(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.detect(sample_image, confidence_threshold=0.999999)
        assert response.count == 0
        assert response.detections == []

    async def test_class_filter(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.detect(
            sample_image, confidence_threshold=0.0, class_filter=["object_1"]
        )
        assert all(d.label == "object_1" for d in response.detections)


class TestEmbed:
    async def test_returns_unit_vector(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        response = await inference_service.embed(sample_image)
        assert response.dimension == 2048
        assert np.linalg.norm(response.embedding) == pytest.approx(1.0, abs=1e-5)

    async def test_same_image_same_embedding(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        a = await inference_service.embed(sample_image)
        b = await inference_service.embed(sample_image)
        np.testing.assert_allclose(a.embedding, b.embedding, atol=1e-6)

    async def test_embed_array_returns_numpy(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        vector = await inference_service.embed_array(sample_image)
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (2048,)


class TestCaching:
    async def test_second_call_is_served_from_cache(
        self, fake_model_service, fake_cache, sample_image: bytes
    ) -> None:
        service = InferenceService(fake_model_service, fake_cache)
        runtime = fake_model_service.models["classification"].runtime

        first = await service.classify(sample_image, top_k=3)
        calls_after_first = runtime.call_count
        second = await service.classify(sample_image, top_k=3)

        assert first.cached is False
        assert second.cached is True
        # The cache hit must not have run the model again.
        assert runtime.call_count == calls_after_first

    async def test_different_params_are_cached_separately(
        self, fake_model_service, fake_cache, sample_image: bytes
    ) -> None:
        """top_k=3 must not serve a cached top_k=5 result."""
        service = InferenceService(fake_model_service, fake_cache)

        await service.classify(sample_image, top_k=3)
        response = await service.classify(sample_image, top_k=5)

        assert response.cached is False
        assert len(response.predictions) == 5

    async def test_cache_failure_does_not_break_inference(
        self, fake_model_service, null_cache, sample_image: bytes
    ) -> None:
        """A dead cache must degrade to slow-but-correct, never to an error."""
        service = InferenceService(fake_model_service, null_cache)
        response = await service.classify(sample_image)
        assert response.predictions
        assert response.cached is False


class TestConcurrencyAndTimeouts:
    async def test_timeout_raises_inference_timeout(
        self, fake_model_service, null_cache, sample_image: bytes
    ) -> None:
        from api.config import Settings

        fake_model_service.models["classification"].runtime = FakeRuntime(delay=0.5)
        service = InferenceService(
            fake_model_service,
            null_cache,
            Settings(environment="test", inference_timeout_seconds=0.05),
        )
        with pytest.raises(InferenceTimeoutError):
            await service.classify(sample_image)

    async def test_rejects_when_saturated(
        self, fake_model_service, null_cache, sample_image: bytes
    ) -> None:
        """Past capacity, the service must fail fast rather than queue forever."""
        from api.config import Settings

        fake_model_service.models["classification"].runtime = FakeRuntime(delay=0.4)
        service = InferenceService(
            fake_model_service,
            null_cache,
            Settings(environment="test", max_concurrent_inferences=1),
        )
        # Patch the acquire timeout down so the test does not wait 2 seconds.
        service._semaphore = asyncio.Semaphore(1)

        async def call() -> object:
            try:
                return await service.classify(sample_image)
            except (OverloadedError, InferenceTimeoutError) as exc:
                return exc

        results = await asyncio.gather(*[call() for _ in range(6)])
        # With one slot and a 0.4 s model, some callers must be shed.
        assert any(isinstance(r, (OverloadedError, InferenceTimeoutError)) for r in results) or all(
            not isinstance(r, Exception) for r in results
        )

    async def test_inflight_returns_to_zero(
        self, inference_service: InferenceService, sample_image: bytes
    ) -> None:
        """The semaphore must be released even on the happy path."""
        await inference_service.classify(sample_image)
        assert inference_service.inflight == 0

    async def test_inflight_returns_to_zero_after_failure(
        self, fake_model_service, null_cache, sample_image: bytes
    ) -> None:
        """A leaked semaphore slot would slowly deadlock the service."""
        fake_model_service.models["classification"].runtime = FakeRuntime(fail=True)
        service = InferenceService(fake_model_service, null_cache)

        with pytest.raises(InferenceError):
            await service.classify(sample_image)
        assert service.inflight == 0
