"""Unit tests for the cache's error handling, reconnection and invalidation.

`test_cache_and_upload_paths.py` covers the happy path and the
never-connected case. This file covers what happens when Redis is there and
then misbehaves, which is the harder half and the half that decides whether a
Redis incident stays a Redis incident.

The governing rule is that **the cache may make the system slower, never
wrong, and never unavailable**. Every operation has a defined answer when
Redis fails: a miss, a False, a zero. None of them raise. So most of these
tests inject a failing client and assert on the return value rather than on an
exception.

The reconnection logic gets particular attention because it is rate-limited.
Retrying on every request during an outage would turn one dead dependency into
a per-request timeout, which is slower than having no cache at all.
"""

from __future__ import annotations

import json
import time

import pytest

from api.config import Settings
from api.services.cache_service import (
    CacheService,
    get_cache_service,
    set_cache_service,
)


class BrokenClient:
    """A Redis client where every call fails."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or ConnectionError("connection refused")
        self.calls: list[str] = []

    def _fail(self, name):
        async def call(*args, **kwargs):
            self.calls.append(name)
            raise self.exc

        return call

    def __getattr__(self, name):
        return self._fail(name)

    def scan_iter(self, *args, **kwargs):
        raise self.exc


def _connected(client) -> CacheService:
    """A service wired to a stub client, as if connect() had succeeded."""
    service = CacheService(Settings(environment="test", cache_enabled=True))
    service._client = client
    service._available = True
    return service


# ---------------------------------------------------------------------------
# Every operation degrades rather than raising
# ---------------------------------------------------------------------------
class TestRedisFailuresDegradeGracefully:
    async def test_get_returns_a_miss(self) -> None:
        service = _connected(BrokenClient())
        assert await service.get("cv:k") is None

    async def test_get_counts_the_miss(self) -> None:
        service = _connected(BrokenClient())
        await service.get("cv:k")
        assert service.stats.misses == 1
        assert service.stats.errors == 1

    async def test_set_reports_false(self) -> None:
        service = _connected(BrokenClient())
        assert await service.set("cv:k", {"a": 1}) is False

    async def test_delete_reports_false(self) -> None:
        service = _connected(BrokenClient())
        assert await service.delete("cv:k") is False

    async def test_invalidate_returns_zero(self) -> None:
        service = _connected(BrokenClient())
        assert await service.invalidate_model("m:1.0.0") == 0

    async def test_health_reports_unavailable(self) -> None:
        service = _connected(BrokenClient())
        health = await service.health()
        assert health["status"] == "unavailable"
        assert health["error"]

    async def test_the_error_is_recorded_for_diagnosis(self) -> None:
        service = _connected(BrokenClient(TimeoutError("timed out")))
        await service.get("cv:k")
        assert "TimeoutError" in service.stats.last_error

    async def test_a_failure_marks_the_cache_unavailable(self) -> None:
        """So subsequent calls skip Redis instead of each paying a timeout."""
        service = _connected(BrokenClient())
        await service.get("cv:k")
        assert service.available is False


# ---------------------------------------------------------------------------
# Reconnection, and its rate limit
# ---------------------------------------------------------------------------
class TestReconnection:
    async def test_recovers_when_redis_comes_back(self, fake_cache) -> None:
        """A cache that never recovers is a cache that is off."""
        client = fake_cache._client
        fake_cache._available = False
        fake_cache._next_retry = 0.0

        await fake_cache._maybe_reconnect()
        assert fake_cache.available is True
        assert client is fake_cache._client

    async def test_does_not_retry_before_the_interval(self) -> None:
        """Retrying per request during an outage is worse than not caching."""
        client = BrokenClient()
        service = _connected(client)
        await service.get("cv:k")  # trips the breaker and sets _next_retry
        attempts_after_first_failure = len(client.calls)

        for _ in range(5):
            await service.get("cv:k")

        assert (
            len(client.calls) == attempts_after_first_failure
        ), "the cache retried Redis on every request during an outage"

    async def test_retries_once_the_interval_has_passed(self) -> None:
        client = BrokenClient()
        service = _connected(client)
        await service.get("cv:k")
        before = len(client.calls)

        service._next_retry = time.monotonic() - 1
        await service.get("cv:k")

        assert len(client.calls) > before

    async def test_a_disabled_cache_never_reconnects(self) -> None:
        service = CacheService(Settings(environment="test", cache_enabled=False))
        service._available = False
        service._next_retry = 0.0
        await service._maybe_reconnect()
        assert service.available is False

    async def test_reconnect_with_no_client_calls_connect(self, monkeypatch) -> None:
        service = CacheService(Settings(environment="test", cache_enabled=True))
        service._available = False
        service._next_retry = 0.0
        service._client = None

        called = {"n": 0}

        async def fake_connect():
            called["n"] += 1
            return False

        monkeypatch.setattr(service, "connect", fake_connect)
        await service._maybe_reconnect()
        assert called["n"] == 1

    async def test_a_failed_reconnect_is_recorded(self) -> None:
        service = _connected(BrokenClient())
        service._available = False
        service._next_retry = 0.0
        await service._maybe_reconnect()
        assert service.available is False
        assert service.stats.errors >= 1


# ---------------------------------------------------------------------------
# Corrupt entries
# ---------------------------------------------------------------------------
class TestCorruptEntries:
    async def test_a_corrupt_entry_is_deleted(self, fake_cache) -> None:
        """One bad write must not poison a key until its TTL expires."""
        await fake_cache._client.set("cv:test:bad", "{not json")
        assert await fake_cache.get("cv:test:bad") is None
        assert await fake_cache._client.get("cv:test:bad") is None

    async def test_a_corrupt_entry_counts_as_a_miss(self, fake_cache) -> None:
        await fake_cache._client.set("cv:test:bad2", "}{")
        before = fake_cache.stats.misses
        await fake_cache.get("cv:test:bad2")
        assert fake_cache.stats.misses == before + 1


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------
class TestInvalidateModel:
    async def test_removes_only_the_named_model(self, fake_cache) -> None:
        await fake_cache.set("cv:classify:doomed:1.0.0:onnx:aaa", {"x": 1})
        await fake_cache.set("cv:classify:doomed:1.0.0:onnx:bbb", {"x": 2})
        await fake_cache.set("cv:classify:keeper:1.0.0:onnx:ccc", {"x": 3})

        removed = await fake_cache.invalidate_model("doomed:1.0.0")

        assert removed == 2
        assert await fake_cache.get("cv:classify:keeper:1.0.0:onnx:ccc") is not None

    async def test_counts_evictions(self, fake_cache) -> None:
        await fake_cache.set("cv:classify:m:1.0.0:onnx:aaa", {"x": 1})
        before = fake_cache.stats.evictions
        await fake_cache.invalidate_model("m:1.0.0")
        assert fake_cache.stats.evictions == before + 1

    async def test_nothing_to_remove_is_not_an_error(self, fake_cache) -> None:
        assert await fake_cache.invalidate_model("never-registered:9.9.9") == 0


# ---------------------------------------------------------------------------
# Small surfaces worth pinning
# ---------------------------------------------------------------------------
class TestSafeUrl:
    def test_a_password_is_redacted(self) -> None:
        service = CacheService(
            Settings(environment="test", redis_url="redis://user:hunter2@cache.internal:6379/0")
        )
        safe = service._safe_url()
        assert "hunter2" not in safe
        assert "cache.internal:6379/0" in safe

    def test_a_url_without_credentials_is_unchanged(self) -> None:
        service = CacheService(Settings(environment="test", redis_url="redis://localhost:6379/0"))
        assert service._safe_url() == "redis://localhost:6379/0"


class TestDisabledCache:
    async def test_health_reports_disabled(self) -> None:
        service = CacheService(Settings(environment="test", cache_enabled=False))
        assert (await service.health())["status"] == "disabled"

    async def test_connect_does_nothing(self) -> None:
        service = CacheService(Settings(environment="test", cache_enabled=False))
        assert await service.connect() is False


class TestClose:
    async def test_close_swallows_a_failing_client(self) -> None:
        """Shutdown must never raise."""
        service = _connected(BrokenClient())
        await service.close()
        assert service.available is False
        assert service._client is None

    async def test_close_clears_availability(self, fake_cache) -> None:
        await fake_cache.close()
        assert fake_cache.available is False


class TestSerialisation:
    async def test_non_json_values_are_stringified_rather_than_failing(self, fake_cache) -> None:
        """`default=str` means a datetime in a payload does not lose the write."""
        from datetime import UTC, datetime

        assert await fake_cache.set("cv:test:dt", {"when": datetime.now(UTC)}) is True
        assert isinstance((await fake_cache.get("cv:test:dt"))["when"], str)

    async def test_stored_json_is_compact(self, fake_cache) -> None:
        await fake_cache.set("cv:test:compact", {"a": 1, "b": 2})
        raw = await fake_cache._client.get("cv:test:compact")
        assert " " not in (raw if isinstance(raw, str) else raw.decode())
        assert json.loads(raw) == {"a": 1, "b": 2}

    async def test_a_custom_ttl_is_applied(self, fake_cache) -> None:
        await fake_cache.set("cv:test:ttl", {"a": 1}, ttl=99)
        assert 0 < await fake_cache._client.ttl("cv:test:ttl") <= 99


class TestSingleton:
    def test_returns_the_same_instance(self) -> None:
        set_cache_service(None)
        assert get_cache_service() is get_cache_service()
        set_cache_service(None)

    def test_can_be_replaced(self) -> None:
        replacement = CacheService()
        set_cache_service(replacement)
        assert get_cache_service() is replacement
        set_cache_service(None)


class TestHealthStatistics:
    async def test_reports_hit_rate(self, fake_cache) -> None:
        await fake_cache.set("cv:test:h", {"a": 1})
        await fake_cache.get("cv:test:h")
        await fake_cache.get("cv:test:absent")
        health = await fake_cache.health()
        assert health["status"] == "healthy"
        assert health["hits"] >= 1
        assert health["misses"] >= 1
        assert health["latency_ms"] >= 0

    async def test_a_ping_failure_during_health_is_reported(self, fake_cache) -> None:
        fake_cache._client = BrokenClient()
        assert (await fake_cache.health())["status"] == "unavailable"


@pytest.mark.parametrize("enabled", [True, False])
async def test_connect_to_an_unreachable_redis(enabled: bool) -> None:
    """Startup must not depend on Redis being up."""
    service = CacheService(
        Settings(
            environment="test",
            cache_enabled=enabled,
            redis_url="redis://127.0.0.1:6397/0",
        )
    )
    assert await service.connect() is False
    assert service.available is False


class TestConnectSucceeds:
    """The happy path of `connect()`, which the other tests bypass.

    Every other test injects a client directly, so the success branch of
    `connect` — ping, mark available, log — is never executed. Here
    `redis.from_url` is pointed at fakeredis so the real function runs.
    """

    async def test_connect_marks_the_cache_available(self, monkeypatch) -> None:
        import fakeredis.aioredis
        import redis.asyncio as redis

        monkeypatch.setattr(
            redis, "from_url", lambda *a, **k: fakeredis.aioredis.FakeRedis(decode_responses=True)
        )
        service = CacheService(Settings(environment="test", cache_enabled=True))

        assert await service.connect() is True
        assert service.available is True
        await service.close()

    async def test_the_service_is_usable_straight_after_connect(self, monkeypatch) -> None:
        import fakeredis.aioredis
        import redis.asyncio as redis

        monkeypatch.setattr(
            redis, "from_url", lambda *a, **k: fakeredis.aioredis.FakeRedis(decode_responses=True)
        )
        service = CacheService(Settings(environment="test", cache_enabled=True))
        await service.connect()

        assert await service.set("cv:test:after-connect", {"a": 1}) is True
        assert (await service.get("cv:test:after-connect"))["a"] == 1
        await service.close()


class TestHealthWhenUnavailable:
    async def test_reports_unavailable_with_statistics(self) -> None:
        """The stats still travel, so a dashboard keeps its hit-rate history."""
        service = CacheService(Settings(environment="test", cache_enabled=True))
        service._available = False
        service._next_retry = time.monotonic() + 3600  # suppress the reconnect attempt

        health = await service.health()
        assert health["status"] == "unavailable"
        assert "hits" in health and "misses" in health
