"""Per-tier rate limiting.

Plain English:
    Rate limiting stops one caller from consuming the whole service. Each
    caller gets an allowance of requests per minute, decided by their
    subscription tier.

We use the **token bucket** algorithm:

    Picture a bucket that holds tokens. Every request takes one token out.
    Tokens refill continuously at a fixed rate. If the bucket is empty, the
    request is rejected.

Why a bucket rather than "count requests in the last minute"? Because a naive
counter that resets on the minute lets a caller fire their entire allowance at
12:00:59 and again at 12:01:00 — double the intended rate in two seconds. A
bucket smooths this out while still permitting a short, deliberate burst.

**Distributed by default.** With several API containers behind a load
balancer, an in-process counter would let a caller get N times their limit by
spreading requests across containers. The bucket therefore lives in Redis and
is updated by a Lua script, which Redis runs atomically — so two containers
cannot both see the last token and both take it.

If Redis is unavailable the limiter fails **open** (allows traffic) rather
than closed. That is a deliberate trade-off: a cache outage should not become
a total outage. It is logged loudly so the gap is visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from api.config import Settings, settings
from api.exceptions import RateLimitError
from api.logging_config import get_logger
from api.middleware.auth import Principal, is_public_path
from api.models.schemas import UserTier

logger = get_logger(__name__)

# Token-bucket update, executed atomically inside Redis.
#
# KEYS[1]  bucket key
# ARGV[1]  capacity (max tokens)
# ARGV[2]  refill rate (tokens per second)
# ARGV[3]  current time (seconds, float)
# ARGV[4]  tokens requested
#
# Returns {allowed (1/0), tokens_remaining, retry_after_seconds}
_TOKEN_BUCKET_LUA = """
local key       = KEYS[1]
local capacity  = tonumber(ARGV[1])
local rate      = tonumber(ARGV[2])
local now       = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'updated')
local tokens = tonumber(bucket[1])
local updated = tonumber(bucket[2])

if tokens == nil then
  tokens = capacity
  updated = now
end

-- Refill based on elapsed time, capped at capacity.
local elapsed = math.max(0, now - updated)
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0

if tokens >= requested then
  allowed = 1
  tokens = tokens - requested
else
  retry_after = (requested - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'updated', now)
-- Expire idle buckets so inactive callers do not accumulate keys forever.
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 60)

return {allowed, tostring(tokens), tostring(retry_after)}
"""  # noqa: S105 - a Lua script, not a credential


@dataclass
class RateLimitDecision:
    """The outcome of one rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after: float
    tier: str

    def headers(self) -> dict[str, str]:
        """Standard ``X-RateLimit-*`` headers to attach to the response.

        Returning these on *every* response, not just rejections, lets a
        well-behaved client slow down before it gets blocked.
        """
        out = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Tier": self.tier,
        }
        if not self.allowed:
            out["Retry-After"] = str(max(1, int(self.retry_after) + 1))
        return out


class _LocalBucket:
    """In-process token bucket, used only when Redis is unavailable.

    Correct for a single process; deliberately not correct across replicas,
    which is why it is a fallback rather than the primary implementation.
    """

    __slots__ = ("tokens", "updated")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.updated = time.monotonic()

    def take(self, capacity: float, rate: float, amount: float = 1.0) -> tuple[bool, float, float]:
        now = time.monotonic()
        self.tokens = min(capacity, self.tokens + (now - self.updated) * rate)
        self.updated = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True, self.tokens, 0.0
        return False, self.tokens, (amount - self.tokens) / rate


class RateLimiter:
    """Token-bucket rate limiter backed by Redis, with a local fallback."""

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or settings
        self._client: Any = None
        self._script: Any = None
        self._available = False
        self._local: dict[str, _LocalBucket] = {}
        self._warned_degraded = False

    async def connect(self) -> bool:
        """Connect to Redis and register the Lua script."""
        if not self.settings.rate_limit_enabled:
            return False
        try:
            import redis.asyncio as redis

            self._client = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            await self._client.ping()
            self._script = self._client.register_script(_TOKEN_BUCKET_LUA)
            self._available = True
            logger.info("rate_limiter_connected")
            return True
        except Exception as exc:
            self._available = False
            logger.warning(
                "rate_limiter_redis_unavailable",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            return False

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
        self._available = False

    def _bucket_params(self, tier: UserTier) -> tuple[float, float]:
        """Return ``(capacity, refill_rate_per_second)`` for a tier."""
        rpm = self.settings.rate_limit_for_tier(tier.value)
        capacity = rpm * self.settings.rate_limit_burst_multiplier
        return float(capacity), rpm / 60.0

    async def check(self, principal: Principal, cost: int = 1) -> RateLimitDecision:
        """Consume ``cost`` tokens for a caller and decide whether to allow them.

        Args:
            principal: The authenticated caller.
            cost: Tokens to consume. Batch endpoints charge more than one, so
                a 50-image batch cannot cost the same as a single image.

        Returns:
            A :class:`RateLimitDecision`; the caller raises if not allowed.
        """
        if not self.settings.rate_limit_enabled:
            return RateLimitDecision(True, 0, 0, 0.0, principal.tier.value)

        capacity, rate = self._bucket_params(principal.tier)
        limit = self.settings.rate_limit_for_tier(principal.tier.value)
        key = f"ratelimit:{principal.tier.value}:{principal.user_id}"

        if self._available and self._script is not None:
            try:
                allowed, tokens, retry_after = await self._script(
                    keys=[key], args=[capacity, rate, time.time(), cost]
                )
                return RateLimitDecision(
                    allowed=bool(int(allowed)),
                    limit=limit,
                    remaining=int(float(tokens)),
                    retry_after=float(retry_after),
                    tier=principal.tier.value,
                )
            except Exception as exc:
                self._available = False
                logger.error(
                    "rate_limiter_failed_open",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )

        # --- Fallback path -------------------------------------------------
        if not self._warned_degraded:
            logger.warning("rate_limiter_degraded_to_local_buckets")
            self._warned_degraded = True

        bucket = self._local.get(key)
        if bucket is None:
            bucket = self._local[key] = _LocalBucket(capacity)
        allowed_b, tokens_f, retry_f = bucket.take(capacity, rate, cost)
        return RateLimitDecision(
            allowed=allowed_b,
            limit=limit,
            remaining=int(tokens_f),
            retry_after=retry_f,
            tier=principal.tier.value,
        )

    async def enforce(self, principal: Principal, cost: int = 1) -> RateLimitDecision:
        """Check the limit and raise :class:`RateLimitError` if exceeded."""
        decision = await self.check(principal, cost)
        if not decision.allowed:
            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "user_id": principal.user_id,
                    "tier": principal.tier.value,
                    "limit_rpm": decision.limit,
                },
            )
            raise RateLimitError(
                (
                    f"You have exceeded the {decision.limit} requests/minute allowance "
                    f"for the '{decision.tier}' tier. Retry in "
                    f"{max(1, int(decision.retry_after) + 1)} seconds."
                ),
                details={
                    "limit_per_minute": decision.limit,
                    "tier": decision.tier,
                    "retry_after_seconds": max(1, int(decision.retry_after) + 1),
                },
            )
        return decision


class RateLimitMiddleware:
    """ASGI middleware applying the rate limit to every authenticated request.

    Runs after :class:`~api.middleware.auth.AuthMiddleware`, so the caller's
    tier is already known.
    """

    def __init__(self, app: Any, limiter: RateLimiter, config: Settings | None = None) -> None:
        self.app = app
        self.limiter = limiter
        self.settings = config or settings

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not self.settings.rate_limit_enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if is_public_path(request.url.path):
            await self.app(scope, receive, send)
            return

        principal: Principal | None = scope.get("state", {}).get("principal")
        if principal is None:
            await self.app(scope, receive, send)
            return

        try:
            decision = await self.limiter.enforce(principal)
        except RateLimitError as exc:
            from api.exceptions import app_error_handler

            response = await app_error_handler(request, exc)
            await response(scope, receive, send)
            return

        # Attach the rate-limit headers to whatever response the app produces.
        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                for name, value in decision.headers().items():
                    headers.append((name.lower().encode(), value.encode()))
            await send(message)

        await self.app(scope, receive, send_with_headers)


# Process-wide singleton, created in the application lifespan handler.
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """FastAPI dependency returning the shared :class:`RateLimiter`."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiter | None) -> None:
    """Replace the singleton. Used by the lifespan handler and by tests."""
    global _rate_limiter
    _rate_limiter = limiter


__all__ = [
    "RateLimitDecision",
    "RateLimitMiddleware",
    "RateLimiter",
    "get_rate_limiter",
    "set_rate_limiter",
]
