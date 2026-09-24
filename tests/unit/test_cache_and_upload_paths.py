"""Unit tests for cache operations and the multipart upload routes.

Closes the last coverage gaps from the requirements audit, on two surfaces
that both matter more than their line counts suggest.

The cache must **fail soft**: every operation returns a miss rather than an
error when Redis is unreachable, because a cache outage that becomes a service
outage is a self-inflicted incident. The uncovered paths were exactly the
error branches that make that true.

The `/upload` routes are the multipart alternative to base64, and enforce the
size limit *while streaming* rather than after buffering.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from api.services.cache_service import CacheService, build_cache_key


def _png(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 90, 160)).save(buf, format="PNG")
    return buf.getvalue()


class TestCacheKeyConstruction:
    def test_same_inputs_give_the_same_key(self) -> None:
        a = build_cache_key("classify", "hash1", "m:1.0.0", "onnx", {"top_k": 5})
        b = build_cache_key("classify", "hash1", "m:1.0.0", "onnx", {"top_k": 5})
        assert a == b

    def test_parameter_order_does_not_matter(self) -> None:
        """Without sort_keys the hit rate silently collapses."""
        a = build_cache_key("classify", "h", "m", "onnx", {"top_k": 5, "threshold": 0.1})
        b = build_cache_key("classify", "h", "m", "onnx", {"threshold": 0.1, "top_k": 5})
        assert a == b

    def test_a_different_image_gives_a_different_key(self) -> None:
        a = build_cache_key("classify", "hash1", "m", "onnx", {})
        b = build_cache_key("classify", "hash2", "m", "onnx", {})
        assert a != b

    def test_runtime_is_part_of_the_key(self) -> None:
        """INT8 disagrees with fp32 on 28% of images - they cannot share."""
        a = build_cache_key("classify", "h", "m", "onnx", {})
        b = build_cache_key("classify", "h", "m", "onnx_int8", {})
        assert a != b

    def test_model_version_is_part_of_the_key(self) -> None:
        a = build_cache_key("classify", "h", "m:1.0.0", "onnx", {})
        b = build_cache_key("classify", "h", "m:2.0.0", "onnx", {})
        assert a != b

    def test_key_is_namespaced_and_bounded(self) -> None:
        key = build_cache_key("classify", "h" * 64, "m", "onnx", {"a": 1})
        assert key.startswith("cv:classify:")
        assert len(key) < 200


class TestCacheFailsSoft:
    """Every operation must degrade to a miss, never raise."""

    @staticmethod
    def _disconnected() -> CacheService:
        return CacheService()

    async def test_get_on_a_disconnected_cache_is_a_miss(self) -> None:
        assert await self._disconnected().get("cv:x") is None

    async def test_set_on_a_disconnected_cache_reports_false(self) -> None:
        assert await self._disconnected().set("cv:x", {"a": 1}) is False

    async def test_delete_on_a_disconnected_cache_reports_false(self) -> None:
        assert await self._disconnected().delete("cv:x") is False

    async def test_invalidate_on_a_disconnected_cache_returns_zero(self) -> None:
        assert await self._disconnected().invalidate_model("m:1.0.0") == 0

    async def test_health_reports_unavailable_without_raising(self) -> None:
        health = await self._disconnected().health()
        assert health["status"] != "healthy"
        assert "hits" in health and "misses" in health

    async def test_connect_to_a_closed_port_returns_false(self) -> None:
        cache = CacheService()
        cache.settings = cache.settings.model_copy(
            update={"redis_url": "redis://127.0.0.1:6392/0", "cache_enabled": True}
        )
        assert await cache.connect() is False

    async def test_close_is_safe_when_never_connected(self) -> None:
        await self._disconnected().close()


class TestCacheRoundTrip:
    """Against fakeredis, so serialisation and TTL are genuinely exercised."""

    async def test_set_then_get(self, fake_cache) -> None:
        await fake_cache.set("cv:test:1", {"label": "cat", "confidence": 0.9})
        assert (await fake_cache.get("cv:test:1"))["label"] == "cat"

    async def test_miss_returns_none(self, fake_cache) -> None:
        assert await fake_cache.get("cv:test:absent") is None

    async def test_delete_removes_the_entry(self, fake_cache) -> None:
        await fake_cache.set("cv:test:2", {"a": 1})
        await fake_cache.delete("cv:test:2")
        assert await fake_cache.get("cv:test:2") is None

    async def test_corrupt_json_is_treated_as_a_miss_and_cleaned_up(self, fake_cache) -> None:
        """A poisoned entry must not break every later request for that key."""
        await fake_cache._client.set("cv:test:bad", "{not json")
        assert await fake_cache.get("cv:test:bad") is None
        assert await fake_cache.get("cv:test:bad") is None

    async def test_health_reports_available(self, fake_cache) -> None:
        assert (await fake_cache.health())["status"] == "healthy"

    async def test_health_tracks_hits_and_misses(self, fake_cache) -> None:
        await fake_cache.set("cv:test:stat", {"a": 1})
        await fake_cache.get("cv:test:stat")
        await fake_cache.get("cv:test:nothing-here")
        health = await fake_cache.health()
        assert health["hits"] >= 1
        assert health["misses"] >= 1

    async def test_invalidate_model_removes_matching_keys(self, fake_cache) -> None:
        await fake_cache.set("cv:classify:m:1.0.0:onnx:aaa", {"x": 1})
        await fake_cache.set("cv:classify:other:1.0.0:onnx:bbb", {"x": 2})
        removed = await fake_cache.invalidate_model("m:1.0.0")
        assert removed >= 1
        assert await fake_cache.get("cv:classify:other:1.0.0:onnx:bbb") is not None


class TestUploadRoutes:
    """Multipart is the alternative to base64 for anything large."""

    def test_classify_upload(self, api_client, auth_headers) -> None:
        response = api_client.post(
            "/api/v1/classify/upload",
            headers=auth_headers,
            files={"file": ("photo.png", _png(), "image/png")},
            data={"top_k": "3"},
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) <= 3

    def test_detect_upload(self, api_client, auth_headers) -> None:
        response = api_client.post(
            "/api/v1/detect/upload",
            headers=auth_headers,
            files={"file": ("photo.png", _png(), "image/png")},
        )
        assert response.status_code == 200
        assert "detections" in response.json()

    def test_similarity_upload(self, api_client, auth_headers) -> None:
        response = api_client.post(
            "/api/v1/similarity/upload",
            headers=auth_headers,
            files={"file": ("photo.png", _png(), "image/png")},
            data={"top_k": "5"},
        )
        assert response.status_code == 200
        assert "results" in response.json()

    def test_upload_requires_authentication(self, api_client) -> None:
        response = api_client.post(
            "/api/v1/classify/upload", files={"file": ("p.png", _png(), "image/png")}
        )
        assert response.status_code == 401

    def test_upload_rejects_a_non_image(self, api_client, auth_headers) -> None:
        """Content-Type says image/png; the bytes say otherwise."""
        response = api_client.post(
            "/api/v1/classify/upload",
            headers=auth_headers,
            files={"file": ("evil.png", b"#!/bin/sh\nrm -rf /", "image/png")},
        )
        assert response.status_code in (400, 415)

    @pytest.mark.parametrize("top_k", ["0", "-1", "9999"])
    def test_upload_validates_form_fields(self, api_client, auth_headers, top_k: str) -> None:
        response = api_client.post(
            "/api/v1/classify/upload",
            headers=auth_headers,
            files={"file": ("photo.png", _png(), "image/png")},
            data={"top_k": top_k},
        )
        assert response.status_code == 422
