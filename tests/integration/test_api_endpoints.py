"""Integration tests for the API endpoints against the real models.

The unit suite drives the same routes with a `FakeRuntime` and `fakeredis`,
which is right for testing routing, validation and error handling quickly.
What it cannot catch is everything that only goes wrong when a real model is
on the other end: an input shape that no longer matches the graph, a label
file out of step with the output layer, a preprocessing preset pointing at the
wrong resolution.

Those failures are quiet. Nothing raises — the API returns 200 with confident
predictions that happen to be wrong, or a silent collapse to near-random
accuracy. So these tests assert on the *content* of responses, not just their
status codes, and the accuracy test is the one that would actually catch a
preprocessing regression.

They skip when the artifacts are absent, so a fresh clone stays green.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytestmark = [pytest.mark.integration, pytest.mark.requires_models, pytest.mark.slow]


@pytest.fixture
def real_client(require_real_models: None, monkeypatch):
    """The real app wired to the real registry, with fakes only for infra.

    Redis and Postgres are pointed at closed ports on purpose: the API is
    supposed to serve predictions without them, and this suite is about the
    models rather than the infrastructure.
    """
    import api.services.cache_service as cache_service
    import api.services.db_service as db_service
    from api.config import Settings
    from api.main import create_app

    unreachable = Settings(
        environment="test",
        redis_url="redis://127.0.0.1:6393/0",
        database_url="postgresql+asyncpg://u:p@127.0.0.1:5493/none",
        auth_enabled=True,
        api_keys="test-pro-key:pro",
        jwt_secret="integration-secret-key-long-enough-for-validation",
        rate_limit_enabled=False,
        eager_model_load=False,
    )
    monkeypatch.setattr(cache_service, "settings", unreachable)
    monkeypatch.setattr(db_service, "settings", unreachable)

    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def headers() -> dict[str, str]:
    return {"X-API-Key": "test-pro-key"}


def _encode(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def photo() -> str:
    """A structured image rather than noise, so predictions are stable."""
    img = Image.new("RGB", (480, 360), (40, 60, 90))
    for x in range(480):
        for y in range(360):
            if (x // 30 + y // 30) % 2 == 0:
                img.putpixel((x, y), (200, 180, 120))
    return _encode(img)


class TestClassification:
    def test_returns_ranked_predictions(self, real_client, headers, photo) -> None:
        response = real_client.post(
            "/api/v1/classify", headers=headers, json={"image_base64": photo, "top_k": 5}
        )
        assert response.status_code == 200, response.text
        predictions = response.json()["predictions"]
        assert len(predictions) == 5

        confidences = [p["confidence"] for p in predictions]
        assert confidences == sorted(confidences, reverse=True), "not ranked"

    def test_labels_are_real_words_not_indices(self, real_client, headers, photo) -> None:
        """A labels file out of step with the model shows up here first."""
        body = real_client.post(
            "/api/v1/classify", headers=headers, json={"image_base64": photo, "top_k": 3}
        ).json()
        labels = [p["label"] for p in body["predictions"]]
        assert all(labels), "empty label"
        assert not all(
            label.startswith("class_") for label in labels
        ), "every label fell back to class_N - the labels file is not loading"

    def test_confidences_are_a_probability_distribution(self, real_client, headers, photo) -> None:
        body = real_client.post(
            "/api/v1/classify",
            headers=headers,
            json={"image_base64": photo, "top_k": 100},
        ).json()
        confidences = [p["confidence"] for p in body["predictions"]]
        assert all(0.0 <= c <= 1.0 for c in confidences)
        assert sum(confidences) <= 1.0 + 1e-3

    def test_top_k_is_honoured(self, real_client, headers, photo) -> None:
        for k in (1, 3, 10):
            body = real_client.post(
                "/api/v1/classify", headers=headers, json={"image_base64": photo, "top_k": k}
            ).json()
            assert len(body["predictions"]) == k

    def test_the_confidence_threshold_filters(self, real_client, headers, photo) -> None:
        body = real_client.post(
            "/api/v1/classify",
            headers=headers,
            json={"image_base64": photo, "top_k": 20, "confidence_threshold": 0.99},
        ).json()
        assert all(p["confidence"] >= 0.99 for p in body["predictions"])

    def test_the_response_names_the_model_that_ran(self, real_client, headers, photo) -> None:
        body = real_client.post(
            "/api/v1/classify", headers=headers, json={"image_base64": photo}
        ).json()
        assert body["model"]["name"]
        assert body["model"]["version"]
        assert body["model"]["runtime"]

    def test_the_same_image_gives_the_same_answer(self, real_client, headers, photo) -> None:
        """Determinism through the whole stack, not just the runtime."""
        first = real_client.post(
            "/api/v1/classify", headers=headers, json={"image_base64": photo, "top_k": 3}
        ).json()["predictions"]
        second = real_client.post(
            "/api/v1/classify", headers=headers, json={"image_base64": photo, "top_k": 3}
        ).json()["predictions"]
        assert [p["label"] for p in first] == [p["label"] for p in second]
        assert first[0]["confidence"] == pytest.approx(second[0]["confidence"])

    def test_a_pinned_model_is_actually_used(self, real_client, headers, photo) -> None:
        response = real_client.post(
            "/api/v1/classify",
            headers=headers,
            json={"image_base64": photo, "model_name": "resnet50-tiny-imagenet", "top_k": 3},
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"]["name"] == "resnet50-tiny-imagenet"

    def test_an_unknown_model_is_404_not_a_substitution(self, real_client, headers, photo) -> None:
        """Silently answering with a different model is worse than an error."""
        response = real_client.post(
            "/api/v1/classify",
            headers=headers,
            json={"image_base64": photo, "model_name": "no-such-model"},
        )
        assert response.status_code == 404


class TestClassifierAccuracy:
    """The test that would catch a preprocessing regression.

    A wrong input size or normalisation does not raise; it collapses accuracy
    to near-random while every status code stays 200. Real labelled images are
    the only thing that notices.
    """

    def test_the_fine_tuned_model_beats_chance_by_a_wide_margin(self, real_client, headers) -> None:
        from pathlib import Path

        pytest.importorskip("models.training.dataset")
        from models.training.dataset import (
            TinyImageNetTrain,
            TinyImageNetVal,
            find_dataset_root,
        )

        try:
            root = find_dataset_root(Path("data"))
        except Exception:
            pytest.skip("Tiny-ImageNet is not present")

        train = TinyImageNetTrain(root)
        val = TinyImageNetVal(root, train.class_to_idx)
        samples = val.samples[:40]

        # The dataset indexes classes by WordNet id ("n01443537"); the API
        # returns the readable name ("goldfish"). The labels file shipped
        # beside the model maps one to the other by position, so comparing
        # through it also checks that the two orderings still agree - which is
        # its own silent failure if the labels file is ever regenerated
        # differently.
        import json as _json

        served_labels = _json.loads(
            Path("models/artifacts/tiny_imagenet_labels.json").read_text(encoding="utf-8")
        )
        assert len(served_labels) == len(train.classes), "labels file and dataset disagree"

        correct = 0
        for path, label in samples:
            encoded = base64.b64encode(path.read_bytes()).decode()
            body = real_client.post(
                "/api/v1/classify",
                headers=headers,
                json={
                    "image_base64": encoded,
                    "model_name": "resnet50-tiny-imagenet",
                    "top_k": 1,
                },
            ).json()
            predicted = body["predictions"][0]["label"]
            if predicted == served_labels[label]:
                correct += 1

        accuracy = correct / len(samples)
        # Random is 0.5% over 200 classes. The model measures ~78%. A floor of
        # 50% is far below the real figure and far above anything a broken
        # preprocessing path could reach by luck.
        assert accuracy > 0.5, (
            f"top-1 {accuracy:.0%} on {len(samples)} real images - "
            "preprocessing or the label mapping is wrong"
        )


class TestDetection:
    def test_returns_a_detections_list(self, real_client, headers, photo) -> None:
        response = real_client.post("/api/v1/detect", headers=headers, json={"image_base64": photo})
        assert response.status_code == 200, response.text
        assert isinstance(response.json()["detections"], list)

    def test_boxes_lie_inside_the_image(self, real_client, headers) -> None:
        """Coordinates outside the frame mean the letterbox maths is wrong.

        This needs an image the detector actually fires on. A synthetic shape
        returns zero detections, which is correct behaviour and makes the
        assertion vacuous: the loop body never runs and the test passes
        whatever the box maths does. Real photos are scanned until one
        produces a detection.
        """
        from pathlib import Path

        try:
            from models.training.dataset import find_dataset_root

            root = find_dataset_root(Path("data"))
        except Exception:
            pytest.skip("Tiny-ImageNet is not present")

        candidates = sorted((root / "val" / "images").glob("*.JPEG"))[:60]
        if not candidates:
            pytest.skip("no validation images found")

        checked = 0
        for path in candidates:
            encoded = base64.b64encode(path.read_bytes()).decode()
            body = real_client.post(
                "/api/v1/detect",
                headers=headers,
                json={"image_base64": encoded, "confidence_threshold": 0.25},
            ).json()

            width, height = body["image_width"], body["image_height"]
            for det in body["detections"]:
                box = det["box"]
                assert 0 <= box["x1"] <= box["x2"] <= width + 1, f"{path.name}: {box}"
                assert 0 <= box["y1"] <= box["y2"] <= height + 1, f"{path.name}: {box}"
                checked += 1

            if checked >= 3:
                break

        assert checked > 0, (
            "no detection was produced across 60 real photos, so the box "
            "geometry was never actually checked"
        )

    def test_max_detections_is_honoured(self, real_client, headers, photo) -> None:
        body = real_client.post(
            "/api/v1/detect",
            headers=headers,
            json={"image_base64": photo, "confidence_threshold": 0.01, "max_detections": 3},
        ).json()
        assert len(body["detections"]) <= 3


class TestSimilarity:
    def test_embedding_has_a_stable_dimension(self, real_client, headers, photo) -> None:
        response = real_client.post(
            "/api/v1/similarity/embed", headers=headers, json={"image_base64": photo}
        )
        assert response.status_code == 200, response.text
        embedding = response.json()["embedding"]
        assert len(embedding) > 0
        assert all(isinstance(v, (int, float)) for v in embedding[:10])

    def test_the_same_image_embeds_identically(self, real_client, headers, photo) -> None:
        first = real_client.post(
            "/api/v1/similarity/embed", headers=headers, json={"image_base64": photo}
        ).json()["embedding"]
        second = real_client.post(
            "/api/v1/similarity/embed", headers=headers, json={"image_base64": photo}
        ).json()["embedding"]
        assert first[:20] == pytest.approx(second[:20])

    def test_indexing_then_searching_finds_the_image(self, real_client, headers, photo) -> None:
        """The round trip the endpoint exists for."""
        indexed = real_client.post(
            "/api/v1/similarity/index",
            headers=headers,
            json={"image_base64": photo, "image_id": "integration-1"},
        )
        assert indexed.status_code in (200, 201), indexed.text

        found = real_client.post(
            "/api/v1/similarity/search",
            headers=headers,
            json={"image_base64": photo, "top_k": 5},
        )
        assert found.status_code == 200
        ids = [r["id"] for r in found.json()["results"]]
        assert "integration-1" in ids

    def test_an_identical_image_scores_near_one(self, real_client, headers, photo) -> None:
        real_client.post(
            "/api/v1/similarity/index",
            headers=headers,
            json={"image_base64": photo, "image_id": "integration-2"},
        )
        results = real_client.post(
            "/api/v1/similarity/search",
            headers=headers,
            json={"image_base64": photo, "top_k": 1},
        ).json()["results"]
        assert results, "nothing came back for an image that was just indexed"
        assert results[0]["score"] > 0.95


class TestModelsEndpoint:
    def test_lists_every_registered_model(self, real_client, headers) -> None:
        body = real_client.get("/api/v1/models", headers=headers).json()
        names = {m["name"] for m in body["models"]}
        assert {"resnet50", "yolov8n", "resnet50-embed"} <= names

    def test_each_entry_carries_its_metadata(self, real_client, headers) -> None:
        body = real_client.get("/api/v1/models", headers=headers).json()
        for model in body["models"]:
            assert model["version"]
            assert model["task"]
            assert model["input_shape"]

    def test_the_input_shape_matches_what_the_model_expects(self, real_client, headers) -> None:
        """Registry metadata drifting from the artifact is how 128 vs 224 hid."""
        from api.services.model_service import ModelService

        service = ModelService()
        body = real_client.get("/api/v1/models", headers=headers).json()
        by_name = {m["name"]: m for m in body["models"]}

        for entry in service.list_entries():
            served = by_name.get(entry.name)
            if served is not None:
                assert served["input_shape"] == list(entry.input_shape)


class TestDegradedInfrastructure:
    """Redis and Postgres are unreachable throughout this module."""

    def test_predictions_still_work(self, real_client, headers, photo) -> None:
        assert (
            real_client.post(
                "/api/v1/classify", headers=headers, json={"image_base64": photo}
            ).status_code
            == 200
        )

    def test_health_says_degraded_rather_than_healthy(self, real_client) -> None:
        body = real_client.get("/api/v1/health").json()
        assert body["status"] in {"degraded", "unhealthy"}

    def test_nothing_is_reported_as_cached(self, real_client, headers, photo) -> None:
        body = real_client.post(
            "/api/v1/classify", headers=headers, json={"image_base64": photo}
        ).json()
        assert body.get("cached") in (False, None)


class TestAuthenticationOnRealEndpoints:
    def test_missing_key_is_rejected(self, real_client, photo) -> None:
        assert real_client.post("/api/v1/classify", json={"image_base64": photo}).status_code == 401

    def test_wrong_key_is_rejected(self, real_client, photo) -> None:
        assert (
            real_client.post(
                "/api/v1/classify",
                headers={"X-API-Key": "not-a-real-key"},
                json={"image_base64": photo},
            ).status_code
            == 401
        )

    def test_health_stays_public(self, real_client) -> None:
        assert real_client.get("/api/v1/health/live").status_code == 200
