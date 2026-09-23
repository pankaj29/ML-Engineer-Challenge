"""Integration tests for the Redis cache.

Run against ``fakeredis``, which implements the real Redis protocol in memory.
That means these exercise genuine serialisation, TTL and key-scanning
behaviour rather than a hand-written mock that agrees with our assumptions.
"""

from __future__ import annotations

import asyncio

import pytest

from api.services.cache_service import CacheService, build_cache_key

pytestmark = pytest.mark.integration


class TestCacheKeys:
    def test_key_is_deterministic(self) -> None:
        a = build_cache_key("classify", "hash1", "resnet50:1.0.0", "onnx", {"top_k": 5})
        b = build_cache_key("classify", "hash1", "resnet50:1.0.0", "onnx", {"top_k": 5})
        assert a == b

    def test_parameter_order_does_not_matter(self) -> None:
        """A dict is unordered; the key must not be."""
        a = build_cache_key("classify", "h", "m:1", "onnx", {"a": 1, "b": 2})
        b = build_cache_key("classify", "h", "m:1", "onnx", {"b": 2, "a": 1})
        assert a == b

    @pytest.mark.parametrize(
        ("field", "changed"),
        [
            ("image", {"image_hash": "different"}),
            ("model", {"model_key": "other:1.0.0"}),
            ("runtime", {"runtime": "onnx_int8"}),
            ("params", {"params": {"top_k": 10}}),
            ("prefix", {"prefix": "detect"}),
        ],
    )
    def test_every_input_changes_the_key(self, field: str, changed: dict) -> None:
        """Forgetting any of these in the key is the classic cache bug."""
        base = {
            "prefix": "classify",
            "image_hash": "hash1",
            "model_key": "resnet50:1.0.0",
            "runtime": "onnx",
            "params": {"top_k": 5},
        }
        assert build_cache_key(**base) != build_cache_key(**{**base, **changed})

    def test_key_is_namespaced(self) -> None:
        key = build_cache_key("classify", "h", "m:1", "onnx", {})
        assert key.startswith("cv:classify:")


class TestCacheOperations:
    async def test_set_then_get(self, fake_cache: CacheService) -> None:
        await fake_cache.set("k1", {"value": 42, "nested": {"a": [1, 2]}})
        assert await fake_cache.get("k1") == {"value": 42, "nested": {"a": [1, 2]}}

    async def test_miss_returns_none(self, fake_cache: CacheService) -> None:
        assert await fake_cache.get("never-written") is None

    async def test_delete(self, fake_cache: CacheService) -> None:
        await fake_cache.set("k2", {"v": 1})
        await fake_cache.delete("k2")
        assert await fake_cache.get("k2") is None

    async def test_ttl_expires_the_entry(self, fake_cache: CacheService) -> None:
        await fake_cache.set("k3", {"v": 1}, ttl=1)
        assert await fake_cache.get("k3") is not None
        await asyncio.sleep(1.1)
        assert await fake_cache.get("k3") is None

    async def test_corrupt_entry_is_treated_as_a_miss(self, fake_cache: CacheService) -> None:
        """A bad write must not poison a key permanently."""
        await fake_cache._client.set("cv:broken", "this is not json{{{")
        assert await fake_cache.get("cv:broken") is None
        # ...and it is cleaned up.
        assert await fake_cache._client.get("cv:broken") is None

    async def test_stats_track_hits_and_misses(self, fake_cache: CacheService) -> None:
        await fake_cache.set("k4", {"v": 1})
        await fake_cache.get("k4")  # hit
        await fake_cache.get("absent")  # miss

        assert fake_cache.stats.hits == 1
        assert fake_cache.stats.misses == 1
        assert fake_cache.stats.hit_rate == 0.5

    async def test_hit_rate_with_no_traffic_is_zero(self) -> None:
        assert CacheService().stats.hit_rate == 0.0


class TestInvalidation:
    async def test_invalidates_only_the_named_model(self, fake_cache: CacheService) -> None:
        """Rolling back one model must not flush the cache for the others."""
        await fake_cache.set(
            build_cache_key("classify", "h1", "modelA:1.0.0", "onnx", {}), {"v": 1}
        )
        await fake_cache.set(
            build_cache_key("classify", "h2", "modelA:1.0.0", "onnx", {}), {"v": 2}
        )
        await fake_cache.set(
            build_cache_key("classify", "h3", "modelB:1.0.0", "onnx", {}), {"v": 3}
        )

        removed = await fake_cache.invalidate_model("modelA:1.0.0")

        assert removed == 2
        assert await fake_cache.get(
            build_cache_key("classify", "h3", "modelB:1.0.0", "onnx", {})
        ) == {"v": 3}

    async def test_invalidating_unknown_model_removes_nothing(
        self, fake_cache: CacheService
    ) -> None:
        await fake_cache.set(build_cache_key("classify", "h", "real:1.0.0", "onnx", {}), {"v": 1})
        assert await fake_cache.invalidate_model("ghost:1.0.0") == 0


class TestGracefulDegradation:
    """A cache outage must degrade to slow, never to broken."""

    async def test_unreachable_redis_does_not_raise(self) -> None:
        from api.config import Settings

        service = CacheService(
            Settings(
                environment="test",
                cache_enabled=True,
                redis_url="redis://127.0.0.1:1/0",  # nothing listens on port 1
            )
        )
        assert await service.connect() is False
        assert await service.get("anything") is None
        assert await service.set("anything", {"v": 1}) is False
        assert (await service.health())["status"] == "unavailable"
        await service.close()

    async def test_disabled_cache_reports_disabled(self) -> None:
        from api.config import Settings

        service = CacheService(Settings(environment="test", cache_enabled=False))
        assert await service.connect() is False
        assert (await service.health())["status"] == "disabled"

    async def test_failure_during_operation_is_absorbed(self, fake_cache: CacheService) -> None:
        """A connection that dies mid-flight must produce a miss, not an error."""

        class ExplodingClient:
            async def get(self, *args, **kwargs):
                raise ConnectionError("connection reset by peer")

            async def set(self, *args, **kwargs):
                raise ConnectionError("connection reset by peer")

            async def ping(self):
                raise ConnectionError("connection reset by peer")

        fake_cache._client = ExplodingClient()
        assert await fake_cache.get("k") is None
        assert await fake_cache.set("k", {"v": 1}) is False
        assert fake_cache.stats.errors >= 1

    async def test_password_is_redacted_in_logs(self) -> None:
        from api.config import Settings

        service = CacheService(
            Settings(environment="test", redis_url="redis://user:supersecret@redis:6379/0")
        )
        assert "supersecret" not in service._safe_url()
        assert "***" in service._safe_url()


class TestHealthReport:
    async def test_healthy_report_has_latency(self, fake_cache: CacheService) -> None:
        health = await fake_cache.health()
        assert health["status"] == "healthy"
        assert health["latency_ms"] >= 0
        assert "hit_rate" in health
