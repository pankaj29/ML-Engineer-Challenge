"""Unit tests for the rate limiter's fallback bucket and middleware wiring.

Part 2 of the brief requires "rate limiting: different limits per user tier".
The audit found `api/middleware/rate_limit.py` at 63% - the untested parts
being precisely the ones that run when Redis is unavailable, and the middleware
that attaches the limit headers.

That matters more than it sounds. The fallback is what runs during an incident,
and a silent, untested fallback already hid a real bug in this project: the
middleware held a limiter that was never connected, so every request used
per-process buckets and limits were multiplied by the replica count.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from api.exceptions import RateLimitError
from api.middleware.auth import Principal
from api.middleware.rate_limit import (
    RateLimitDecision,
    RateLimiter,
    _LocalBucket,
)
from api.models.schemas import UserTier


class TestLocalBucket:
    """The in-process fallback. Correct for one process, by design."""

    def test_starts_full(self) -> None:
        bucket = _LocalBucket(capacity=10.0)
        allowed, remaining, retry = bucket.take(capacity=10.0, rate=1.0)
        assert allowed
        assert remaining == pytest.approx(9.0, abs=0.01)
        assert retry == 0.0

    def test_empties_after_enough_requests(self) -> None:
        bucket = _LocalBucket(capacity=3.0)
        for _ in range(3):
            assert bucket.take(capacity=3.0, rate=0.001)[0]
        allowed, _, retry = bucket.take(capacity=3.0, rate=0.001)
        assert not allowed
        assert retry > 0

    def test_a_never_refilling_bucket_does_not_divide_by_zero(self) -> None:
        """rate=0 blocks a tier outright (RATE_LIMIT_FREE_RPM=0).

        Regression test: this used to raise ZeroDivisionError, turning a
        deliberate configuration into a 500.
        """
        bucket = _LocalBucket(capacity=1.0)
        assert bucket.take(capacity=1.0, rate=0.0)[0]
        allowed, _, retry = bucket.take(capacity=1.0, rate=0.0)
        assert not allowed
        assert retry == float("inf")

    def test_refills_over_time(self) -> None:
        bucket = _LocalBucket(capacity=2.0)
        bucket.take(capacity=2.0, rate=1000.0)
        bucket.take(capacity=2.0, rate=1000.0)
        time.sleep(0.01)  # at 1000 tokens/sec this is plenty
        allowed, _, _ = bucket.take(capacity=2.0, rate=1000.0)
        assert allowed

    def test_never_refills_above_capacity(self) -> None:
        """An idle client gets a burst, but not an unlimited one."""
        bucket = _LocalBucket(capacity=5.0)
        time.sleep(0.01)
        _, remaining, _ = bucket.take(capacity=5.0, rate=1_000_000.0)
        assert remaining <= 5.0

    def test_cost_greater_than_one_is_honoured(self) -> None:
        """A 50-image batch must not cost the same as one image."""
        bucket = _LocalBucket(capacity=10.0)
        allowed, remaining, _ = bucket.take(capacity=10.0, rate=0.0, amount=4.0)
        assert allowed
        assert remaining == pytest.approx(6.0, abs=0.01)


class TestRateLimitDecision:
    def test_headers_are_strings_and_complete(self) -> None:
        decision = RateLimitDecision(
            allowed=True, limit=300, remaining=299, retry_after=0.0, tier="pro"
        )
        headers = decision.headers()
        assert headers["X-RateLimit-Limit"] == "300"
        assert headers["X-RateLimit-Remaining"] == "299"
        assert headers["X-RateLimit-Tier"] == "pro"
        assert all(isinstance(v, str) for v in headers.values())

    def test_remaining_never_reported_negative(self) -> None:
        decision = RateLimitDecision(
            allowed=False, limit=10, remaining=-3, retry_after=5.0, tier="free"
        )
        assert int(decision.headers()["X-RateLimit-Remaining"]) >= 0


class TestLimiterWithoutRedis:
    """With no Redis connected, the limiter degrades rather than failing."""

    @staticmethod
    def _limiter() -> RateLimiter:
        """A limiter with limiting switched ON.

        conftest sets RATE_LIMIT_ENABLED=false for the suite, so a default
        limiter short-circuits and allows everything - which would make every
        assertion below vacuously true.
        """
        limiter = RateLimiter()
        limiter.settings = limiter.settings.model_copy(update={"rate_limit_enabled": True})
        return limiter

    @staticmethod
    def _principal(tier: UserTier = UserTier.FREE) -> Principal:
        return Principal(user_id="u-fallback", tier=tier, auth_method="api_key")

    async def test_allows_when_redis_was_never_connected(self) -> None:
        """Fails OPEN. A limiter outage must not become a service outage."""
        limiter = self._limiter()
        decision = await limiter.check(self._principal())
        assert decision.allowed

    async def test_reports_the_tier_it_applied(self) -> None:
        limiter = self._limiter()
        decision = await limiter.check(self._principal(UserTier.ENTERPRISE))
        assert decision.tier == "enterprise"

    async def test_higher_tiers_get_higher_limits(self) -> None:
        limiter = self._limiter()
        free = await limiter.check(self._principal(UserTier.FREE))
        pro = await limiter.check(self._principal(UserTier.PRO))
        assert pro.limit > free.limit

    async def test_free_tier_is_exhaustible(self) -> None:
        """The fallback still limits - it just does so per process."""
        limiter = self._limiter()
        principal = Principal(user_id="u-burner", tier=UserTier.FREE, auth_method="api_key")
        outcomes = [(await limiter.check(principal)).allowed for _ in range(400)]
        assert not all(outcomes), "free tier should run out within 400 requests"

    async def test_enforce_raises_once_the_bucket_is_empty(self) -> None:
        limiter = self._limiter()
        principal = Principal(user_id="u-enforce", tier=UserTier.FREE, auth_method="api_key")
        with pytest.raises(RateLimitError):
            for _ in range(500):
                await limiter.enforce(principal)

    async def test_separate_users_have_separate_buckets(self) -> None:
        limiter = self._limiter()
        a = Principal(user_id="user-a", tier=UserTier.FREE, auth_method="api_key")
        b = Principal(user_id="user-b", tier=UserTier.FREE, auth_method="api_key")
        for _ in range(300):
            await limiter.check(a)
        assert (await limiter.check(b)).allowed, "user-b starved by user-a's traffic"

    async def test_connect_returns_false_rather_than_raising(self) -> None:
        """An unreachable Redis is reported, not thrown."""
        limiter = RateLimiter()
        limiter.settings = limiter.settings.model_copy(
            update={"redis_url": "redis://127.0.0.1:6390/0", "rate_limit_enabled": True}
        )
        assert await asyncio.wait_for(limiter.connect(), timeout=15) is False


class TestDisabledLimiter:
    async def test_disabled_limiter_always_allows(self) -> None:
        limiter = RateLimiter()
        limiter.settings = limiter.settings.model_copy(update={"rate_limit_enabled": False})
        principal = Principal(user_id="u", tier=UserTier.FREE, auth_method="api_key")
        for _ in range(50):
            assert (await limiter.check(principal)).allowed
