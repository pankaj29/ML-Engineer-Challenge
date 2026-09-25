"""The sample images must behave the way the documentation says they do.

`samples/README.md` publishes a table of what each image classifies and
detects as, and the main README tells a reader that `samples/dog.jpg` gives a
Labrador retriever and a dog. Those are claims a reader checks in their first
two minutes, so a model swap that quietly invalidates them is worth catching
here rather than in someone's first impression.

Thresholds are loose. The point is that the documented label still wins, not
that the confidence is unchanged.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_models]

SAMPLES = Path("samples")


@pytest.fixture
def client(require_real_models: None, monkeypatch):
    from fastapi.testclient import TestClient

    import api.services.cache_service as cache_service
    import api.services.db_service as db_service
    from api.config import Settings
    from api.main import create_app

    unreachable = Settings(
        environment="test",
        redis_url="redis://127.0.0.1:6394/0",
        database_url="postgresql+asyncpg://u:p@127.0.0.1:5494/none",
        auth_enabled=True,
        api_keys="test-pro-key:pro",
        jwt_secret="documented-samples-secret-long-enough-to-validate",
        rate_limit_enabled=False,
        eager_model_load=False,
    )
    monkeypatch.setattr(cache_service, "settings", unreachable)
    monkeypatch.setattr(db_service, "settings", unreachable)
    with TestClient(create_app()) as test_client:
        yield test_client


HEADERS = {"X-API-Key": "test-pro-key"}


def encoded(name: str) -> str:
    return base64.b64encode((SAMPLES / name).read_bytes()).decode()


class TestTheSampleImagesExist:
    @pytest.mark.parametrize("name", ["dog.jpg", "street.jpg", "person.jpg"])
    def test_present_and_a_real_image(self, name: str) -> None:
        path = SAMPLES / name
        assert path.is_file(), f"{path} is referenced by the docs but missing"
        assert path.stat().st_size > 10_000, "suspiciously small for a photo"
        assert path.read_bytes()[:3] == b"\xff\xd8\xff", "not a JPEG"

    def test_provenance_is_recorded(self) -> None:
        """Committed third-party images need their source on file."""
        sources = json.loads((SAMPLES / "SOURCES.json").read_text(encoding="utf-8"))
        for name in ("dog.jpg", "street.jpg", "person.jpg"):
            assert name in sources
            assert sources[name]["author"]
            assert sources[name]["url"].startswith("https://")

    def test_they_are_not_git_lfs_pointers(self) -> None:
        for name in ("dog.jpg", "street.jpg", "person.jpg"):
            head = (SAMPLES / name).read_bytes()[:64]
            assert b"git-lfs" not in head, f"{name} is an unfetched LFS pointer"


class TestDocumentedClassifications:
    def test_dog_classifies_as_a_labrador(self, client) -> None:
        """The README's headline example."""
        body = client.post(
            "/api/v1/classify",
            headers=HEADERS,
            json={"image_base64": encoded("dog.jpg"), "top_k": 5},
        ).json()
        labels = [p["label"] for p in body["predictions"]]
        assert "Labrador retriever" in labels, f"README says Labrador retriever, got {labels}"

    def test_street_classifies_as_a_vehicle(self, client) -> None:
        body = client.post(
            "/api/v1/classify",
            headers=HEADERS,
            json={"image_base64": encoded("street.jpg"), "top_k": 10},
        ).json()
        labels = " ".join(p["label"] for p in body["predictions"]).lower()
        assert any(
            word in labels for word in ("convertible", "car", "truck", "jeep", "cab", "wagon")
        ), f"expected something vehicle-shaped, got {labels}"


class TestDocumentedDetections:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("dog.jpg", "dog"), ("street.jpg", "truck"), ("person.jpg", "person")],
    )
    def test_the_documented_object_is_found(self, client, name: str, expected: str) -> None:
        body = client.post(
            "/api/v1/detect",
            headers=HEADERS,
            json={"image_base64": encoded(name), "confidence_threshold": 0.25},
        ).json()
        found = [d["label"] for d in body["detections"]]
        assert expected in found, f"samples/README.md says {expected}, got {found}"

    def test_boxes_are_inside_the_image(self, client) -> None:
        """A real detection, so the geometry is genuinely checked."""
        body = client.post(
            "/api/v1/detect",
            headers=HEADERS,
            json={"image_base64": encoded("dog.jpg"), "confidence_threshold": 0.25},
        ).json()
        width, height = body["image_width"], body["image_height"]
        assert body["detections"], "dog.jpg must produce a detection for this to mean anything"
        for det in body["detections"]:
            box = det["box"]
            assert 0 <= box["x1"] <= box["x2"] <= width + 1
            assert 0 <= box["y1"] <= box["y2"] <= height + 1


class TestPersonIsAnHonestFailureCase:
    def test_the_classifier_has_no_person_class(self, client) -> None:
        """samples/README.md uses this to show a confident label is not
        evidence the model understood the picture."""
        body = client.post(
            "/api/v1/classify",
            headers=HEADERS,
            json={"image_base64": encoded("person.jpg"), "top_k": 5},
        ).json()
        labels = [p["label"].lower() for p in body["predictions"]]
        assert "person" not in labels, "ImageNet-1k has no person class"

    def test_but_the_detector_finds_them(self, client) -> None:
        body = client.post(
            "/api/v1/detect",
            headers=HEADERS,
            json={"image_base64": encoded("person.jpg"), "confidence_threshold": 0.25},
        ).json()
        assert "person" in [d["label"] for d in body["detections"]]
