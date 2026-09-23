"""Redis-backed result caching.

Plain English:
    Running a model costs real CPU time. If the same image is sent twice, we
    should answer the second time from memory instead of paying that cost
    again. That is all this service does — but doing it safely takes care.

Design decisions worth knowing:

* **The cache key is a hash of the image bytes plus every parameter that
  changes the answer** (model name, version, runtime, top_k, thresholds).
  Forgetting a parameter is the classic cache bug: a user asks for top_k=5,
  then top_k=20, and gets the cached 5 back.
* **The cache never breaks the API.** Every Redis call is wrapped so that a
  dead cache degrades to "slow but correct" rather than "500 error". This is
  the single most important property here: a cache is an optimisation, and an
  optimisation must never become a dependency.
* **Connections are created once**, at startup, and reused. Opening a TCP
  connection per request would cost more than the cache saves.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from api.config import Settings, settings
from api.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Counters for cache effectiveness, exposed on the health endpoint."""

    hits: int = 0
    misses: int = 0
    errors: int = 0
    sets: int = 0
    evictions: int = 0
    last_error: str | None = field(default=None)

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache. 0.0 when nothing was asked."""
        return self.hits / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "sets": self.sets,
            "hit_rate": round(self.hit_rate, 4),
            "last_error": self.last_error,
        }


def build_cache_key(
    prefix: str,
    image_hash: str,
    model_key: str,
    runtime: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Build a deterministic cache key.

    Every input that can change the result must be part of the key. The
    parameter dictionary is serialised with sorted keys so that ``{"a":1,
    "b":2}`` and ``{"b":2, "a":1}`` produce the same key.

    Args:
        prefix: Namespace, usually the task name (``"classify"``).
        image_hash: SHA-256 of the raw image bytes.
        model_key: ``name:version`` of the model used.
        runtime: Runtime format, since INT8 and float32 give different answers.
        params: Task parameters such as ``top_k`` or ``iou_threshold``.

    Returns:
        A key of the form ``cv:classify:<model>:<runtime>:<digest>``.
    """
    payload = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{image_hash}|{payload}".encode()).hexdigest()[:32]
    return f"cv:{prefix}:{model_key}:{runtime}:{digest}"


def _record(operation: str, result: str) -> None:
    """Publish a cache outcome to Prometheus.

    Imported lazily and wrapped: a metrics problem must never be able to break
    a cache operation, which in turn must never break a request.
    """
    try:
        from api.middleware.monitoring import record_cache

        record_cache(operation, result)
    except Exception:  # pragma: no cover - metrics are strictly best-effort
        pass


class CacheService:
    """Async Redis cache with graceful failure.

    If Redis is unreachable the service records the error, reports itself
    unhealthy, and returns cache misses — the API keeps working at full
    correctness, just without the speed-up.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or settings
        self.stats = CacheStats()
        self._client: Any = None
        self._available = False
        # After a connection failure, stop hammering Redis on every request.
        # We retry at most once every `_retry_interval` seconds.
        self._retry_interval = 10.0
        self._next_retry = 0.0

    # ------------------------------------------------------------ lifecycle --
    async def connect(self) -> bool:
        """Open the connection pool. Safe to call when Redis is down."""
        if not self.settings.cache_enabled:
            logger.info("cache_disabled_by_config")
            return False
        try:
            import redis.asyncio as redis

            self._client = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
                health_check_interval=30,
                retry_on_timeout=True,
            )
            await self._client.ping()
            self._available = True
            logger.info("cache_connected", extra={"url": self._safe_url()})
            return True
        except Exception as exc:
            self._available = False
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "cache_unavailable_at_startup",
                extra={"url": self._safe_url(), "error": self.stats.last_error},
            )
            return False

    async def close(self) -> None:
        """Close the connection pool during shutdown."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # shutdown must never raise
                pass
            self._client = None
        self._available = False

    def _safe_url(self) -> str:
        """Redis URL with any password removed, safe to log."""
        url = self.settings.redis_url
        if "@" in url:
            scheme, _, rest = url.partition("://")
            _, _, host = rest.rpartition("@")
            return f"{scheme}://***@{host}"
        return url

    @property
    def available(self) -> bool:
        """True when the cache is believed to be usable."""
        return self._available and self._client is not None

    def _record_failure(self, exc: Exception) -> None:
        """Mark the cache as down and schedule a retry."""
        self.stats.errors += 1
        self.stats.last_error = f"{type(exc).__name__}: {exc}"
        self._available = False
        self._next_retry = time.monotonic() + self._retry_interval

    async def _maybe_reconnect(self) -> None:
        """Try to come back after a failure, but not more than once per interval."""
        if self._available or not self.settings.cache_enabled:
            return
        if time.monotonic() < self._next_retry:
            return
        self._next_retry = time.monotonic() + self._retry_interval
        if self._client is None:
            await self.connect()
            return
        try:
            await self._client.ping()
            self._available = True
            logger.info("cache_recovered")
        except Exception as exc:
            self._record_failure(exc)

    # -------------------------------------------------------------- read/write --
    async def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached result. Returns ``None`` on a miss or any error."""
        await self._maybe_reconnect()
        if not self.available:
            self.stats.misses += 1
            _record("get", "unavailable")
            return None
        try:
            raw = await self._client.get(key)
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("cache_get_failed", extra={"error": self.stats.last_error})
            self.stats.misses += 1
            _record("get", "error")
            return None

        if raw is None:
            self.stats.misses += 1
            _record("get", "miss")
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            # A corrupt entry is treated as a miss and deleted, so one bad
            # write cannot poison a key permanently.
            self.stats.misses += 1
            _record("get", "corrupt")
            await self.delete(key)
            return None

        self.stats.hits += 1
        _record("get", "hit")
        return value

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> bool:
        """Store a result. Returns False if it could not be cached."""
        await self._maybe_reconnect()
        if not self.available:
            return False
        try:
            await self._client.set(
                key,
                json.dumps(value, separators=(",", ":"), default=str),
                ex=ttl or self.settings.cache_ttl_seconds,
            )
            self.stats.sets += 1
            _record("set", "ok")
            return True
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("cache_set_failed", extra={"error": self.stats.last_error})
            _record("set", "error")
            return False

    async def delete(self, key: str) -> bool:
        """Remove one key."""
        if not self.available:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception as exc:
            self._record_failure(exc)
            return False

    async def invalidate_model(self, model_key: str) -> int:
        """Drop every cached result produced by one model version.

        Called after a model is re-registered or rolled back, so that stale
        predictions from the old weights cannot be served.

        ``scan_iter`` is used rather than ``KEYS`` because ``KEYS`` blocks the
        entire Redis server while it walks the keyspace — fine with 100 keys,
        a production outage with 10 million.
        """
        if not self.available:
            return 0
        removed = 0
        try:
            async for key in self._client.scan_iter(match=f"cv:*:{model_key}:*", count=500):
                await self._client.delete(key)
                removed += 1
        except Exception as exc:
            self._record_failure(exc)
            logger.warning("cache_invalidate_failed", extra={"error": self.stats.last_error})
        if removed:
            self.stats.evictions += removed
            logger.info("cache_invalidated", extra={"model": model_key, "keys_removed": removed})
        return removed

    # --------------------------------------------------------------- health --
    async def health(self) -> dict[str, Any]:
        """Probe Redis and report status plus hit-rate statistics."""
        if not self.settings.cache_enabled:
            return {"status": "disabled", **self.stats.to_dict()}

        await self._maybe_reconnect()
        if not self.available:
            return {
                "status": "unavailable",
                "error": self.stats.last_error,
                **self.stats.to_dict(),
            }

        started = time.perf_counter()
        try:
            await self._client.ping()
            latency_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            self._record_failure(exc)
            return {"status": "unavailable", "error": self.stats.last_error, **self.stats.to_dict()}

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            **self.stats.to_dict(),
        }


# Process-wide singleton, created in the application lifespan handler.
_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """FastAPI dependency returning the shared :class:`CacheService`."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service


def set_cache_service(service: CacheService | None) -> None:
    """Replace the singleton. Used by the lifespan handler and by tests."""
    global _cache_service
    _cache_service = service


__all__ = [
    "CacheService",
    "CacheStats",
    "build_cache_key",
    "get_cache_service",
    "set_cache_service",
]
