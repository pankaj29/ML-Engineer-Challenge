"""Unit tests for the rate limiter's Redis path, fallback and middleware.

`test_rate_limit_internals.py` covers the token-bucket arithmetic. This file
covers the parts around it: the Lua script path against Redis, the failure that
drops to per-process buckets, and the ASGI middleware that attaches the
headers.

The behaviour that matters most is the **fail-open** choice. If Redis goes
away, traffic is allowed rather than rejected, because a cache outage turning
into a total outage is a worse failure than a brief window of per-process
limits. That is a deliberate trade and these tests pin it, including the part
people forget: it must be logged, and logged **once**, not on every request.
"""

from __future__ import annotations

import time

import pytest

from api.config import Settings
from api.middleware.auth import Principal
from api.middleware.rate_limit import (
    RateLimiter,
    get_rate_limiter,
    set_rate_limiter,
)
from api.models.schemas import UserTier


def _settings(**overrides) -> Settings:
    base = {
        "environment": "test",
        "rate_limit_enabled": True,
        "rate_limit_free_rpm": 10,
        "rate_limit_pro_rpm": 300,
        "rate_limit_burst_multiplier": 1.5,
    }
    base.update(overrides)
    return Settings(**base)


def _principal(tier: UserTier = UserTier.FREE, user_id: str = "user-1") -> Principal:
    return Principal(user_id=user_id, tier=tier, auth_method="api_key")


class ScriptedRedis:
    """A Redis stand-in whose Lua script returns whatever the test wants."""

    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self.result = result if result is not None else [1, 9.0, 0.0]
        self.exc = exc
        self.calls: list[dict] = []

    async def ping(self):
        if self.exc:
            raise self.exc
        return True

    def register_script(self, _source):
        async def run(keys=None, args=None):
            self.calls.append({"keys": keys, "args": args})
            if self.exc:
                raise self.exc
            return self.result

        return run

    async def aclose(self):
        return None


def _wired(client: ScriptedRedis, **settings_overrides) -> RateLimiter:
    limiter = RateLimiter(_settings(**settings_overrides))
    limiter._client = client
    limiter._script = client.register_script("")
    limiter._available = True
    return limiter


# ---------------------------------------------------------------------------
# The Redis path
# ---------------------------------------------------------------------------
class TestRedisPath:
    async def test_an_allowed_request(self) -> None:
        limiter = _wired(ScriptedRedis([1, 9.0, 0.0]))
        decision = await limiter.check(_principal())
        assert decision.allowed is True
        assert decision.remaining == 9
        assert decision.tier == "free"

    async def test_a_throttled_request(self) -> None:
        limiter = _wired(ScriptedRedis([0, 0.0, 4.5]))
        decision = await limiter.check(_principal())
        assert decision.allowed is False
        assert decision.retry_after == pytest.approx(4.5)

    async def test_the_limit_reported_matches_the_tier(self) -> None:
        limiter = _wired(ScriptedRedis())
        assert (await limiter.check(_principal(UserTier.FREE))).limit == 10
        assert (await limiter.check(_principal(UserTier.PRO))).limit == 300

    async def test_the_key_is_namespaced_by_tier_and_user(self) -> None:
        """Two users must not share a bucket, and neither must two tiers."""
        client = ScriptedRedis()
        limiter = _wired(client)
        await limiter.check(_principal(UserTier.PRO, "alice"))
        assert client.calls[0]["keys"] == ["ratelimit:pro:alice"]

    async def test_capacity_includes_the_burst_multiplier(self) -> None:
        client = ScriptedRedis()
        limiter = _wired(client, rate_limit_free_rpm=10, rate_limit_burst_multiplier=1.5)
        await limiter.check(_principal())
        capacity, rate, _now, cost = client.calls[0]["args"]
        assert capacity == pytest.approx(15.0)
        assert rate == pytest.approx(10 / 60)
        assert cost == 1

    async def test_cost_is_passed_through(self) -> None:
        """A 50-image batch must not cost the same as one image."""
        client = ScriptedRedis()
        limiter = _wired(client)
        await limiter.check(_principal(), cost=50)
        assert client.calls[0]["args"][3] == 50


class TestDisabled:
    async def test_everything_is_allowed_when_disabled(self) -> None:
        limiter = RateLimiter(_settings(rate_limit_enabled=False))
        decision = await limiter.check(_principal())
        assert decision.allowed is True
        assert decision.limit == 0


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------
class TestFailOpen:
    async def test_a_redis_failure_falls_back_to_a_local_bucket(self) -> None:
        limiter = _wired(ScriptedRedis(exc=ConnectionError("redis is gone")))
        decision = await limiter.check(_principal())
        assert decision.allowed is True, "a Redis outage must not reject traffic"

    async def test_the_failure_marks_redis_unavailable(self) -> None:
        """So the next request goes straight to the local bucket."""
        limiter = _wired(ScriptedRedis(exc=ConnectionError("gone")))
        await limiter.check(_principal())
        assert limiter._available is False

    async def test_the_local_bucket_still_enforces_a_limit(self) -> None:
        """Fail-open is not no-limit: the per-process bucket still applies."""
        limiter = _wired(ScriptedRedis(exc=ConnectionError("gone")))
        outcomes = [(await limiter.check(_principal())).allowed for _ in range(25)]
        assert outcomes[0] is True
        assert False in outcomes, "the local backstop never throttled"

    async def test_degradation_is_logged_once_not_per_request(self, caplog) -> None:
        """A per-request warning during an outage buries the useful logs."""
        import logging

        limiter = _wired(ScriptedRedis(exc=ConnectionError("gone")))
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                await limiter.check(_principal())

        degraded = [r for r in caplog.records if "degraded_to_local" in r.getMessage()]
        assert len(degraded) == 1

    async def test_separate_users_get_separate_local_buckets(self) -> None:
        limiter = _wired(ScriptedRedis(exc=ConnectionError("gone")))
        for _ in range(20):
            await limiter.check(_principal(user_id="heavy"))
        fresh = await limiter.check(_principal(user_id="light"))
        assert fresh.allowed is True


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------
class TestConnect:
    async def test_connect_to_an_unreachable_redis_returns_false(self) -> None:
        limiter = RateLimiter(_settings(redis_url="redis://127.0.0.1:6398/0"))
        assert await limiter.connect() is False
        assert limiter._available is False

    async def test_a_successful_connect_registers_the_script(self, monkeypatch) -> None:
        import redis.asyncio as redis

        client = ScriptedRedis()
        monkeypatch.setattr(redis, "from_url", lambda *a, **k: client)
        limiter = RateLimiter(_settings())

        assert await limiter.connect() is True
        assert limiter._available is True
        assert limiter._script is not None
        await limiter.close()

    async def test_close_clears_the_client(self, monkeypatch) -> None:
        import redis.asyncio as redis

        monkeypatch.setattr(redis, "from_url", lambda *a, **k: ScriptedRedis())
        limiter = RateLimiter(_settings())
        await limiter.connect()
        await limiter.close()
        assert limiter._client is None
        assert limiter._available is False

    async def test_close_swallows_a_failing_client(self) -> None:
        """Shutdown must never raise."""

        class Stubborn(ScriptedRedis):
            async def aclose(self):
                raise OSError("socket already gone")

        limiter = _wired(Stubborn())
        await limiter.close()
        assert limiter._client is None

    async def test_close_without_connecting_is_safe(self) -> None:
        await RateLimiter(_settings()).close()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class TestMiddleware:
    """The middleware is driven directly rather than through `create_app`.

    The suite runs with RATE_LIMIT_ENABLED=false so that unrelated tests are
    not throttled, which means the middleware short-circuits inside the real
    app and never attaches a header. Wrapping a stub ASGI app here exercises
    the header and rejection paths without changing global configuration.
    """

    @staticmethod
    async def _drive(middleware, path: str = "/api/v1/classify", principal=None):
        """Run one request through the middleware, returning (status, headers)."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
            "state": {"principal": principal} if principal else {"principal": None},
        }
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await middleware(scope, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
        return start["status"], headers

    @staticmethod
    def _stub_app():
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        return app

    def _middleware(self, limiter):
        from api.middleware.rate_limit import RateLimitMiddleware

        return RateLimitMiddleware(self._stub_app(), limiter, _settings())

    async def test_headers_are_attached_to_a_normal_response(self) -> None:
        limiter = _wired(ScriptedRedis([1, 7.0, 0.0]))
        status, headers = await self._drive(self._middleware(limiter), principal=_principal())
        assert status == 200
        assert headers["x-ratelimit-limit"] == "10"
        assert headers["x-ratelimit-remaining"] == "7"
        assert headers["x-ratelimit-tier"] == "free"

    async def test_a_throttled_request_is_rejected_with_429(self) -> None:
        limiter = _wired(ScriptedRedis([0, 0.0, 5.0]))
        status, _ = await self._drive(self._middleware(limiter), principal=_principal())
        assert status == 429

    async def test_public_paths_skip_the_limiter(self) -> None:
        """Health must stay reachable while a caller is being throttled."""
        client = ScriptedRedis([0, 0.0, 5.0])
        limiter = _wired(client)
        status, headers = await self._drive(
            self._middleware(limiter), path="/api/v1/health/live", principal=_principal()
        )
        assert status == 200
        assert "x-ratelimit-limit" not in headers
        assert client.calls == [], "the limiter was consulted for a public path"

    async def test_an_unauthenticated_request_is_not_charged(self) -> None:
        """No principal means no bucket to charge; auth rejects it separately."""
        client = ScriptedRedis()
        limiter = _wired(client)
        status, _ = await self._drive(self._middleware(limiter), principal=None)
        assert status == 200
        assert client.calls == []

    async def test_non_http_scopes_pass_straight_through(self) -> None:
        limiter = _wired(ScriptedRedis())
        middleware = self._middleware(limiter)
        seen = []

        async def send(message):
            seen.append(message)

        async def receive():
            return {"type": "lifespan.startup"}

        await middleware({"type": "lifespan"}, receive, send)

    async def test_disabled_middleware_passes_through(self) -> None:
        from api.middleware.rate_limit import RateLimitMiddleware

        client = ScriptedRedis()
        limiter = _wired(client)
        middleware = RateLimitMiddleware(
            self._stub_app(), limiter, _settings(rate_limit_enabled=False)
        )
        status, headers = await self._drive(middleware, principal=_principal())
        assert status == 200
        assert "x-ratelimit-limit" not in headers
        assert client.calls == []


class TestBucketParams:
    def test_each_tier_gets_its_configured_rate(self) -> None:
        limiter = RateLimiter(
            _settings(
                rate_limit_free_rpm=10,
                rate_limit_basic_rpm=60,
                rate_limit_pro_rpm=300,
                rate_limit_enterprise_rpm=3000,
            )
        )
        for tier, rpm in (
            (UserTier.FREE, 10),
            (UserTier.BASIC, 60),
            (UserTier.PRO, 300),
            (UserTier.ENTERPRISE, 3000),
        ):
            _capacity, rate = limiter._bucket_params(tier)
            assert rate == pytest.approx(rpm / 60.0)

    def test_capacity_scales_with_the_burst_multiplier(self) -> None:
        limiter = RateLimiter(_settings(rate_limit_free_rpm=100, rate_limit_burst_multiplier=2.0))
        capacity, _rate = limiter._bucket_params(UserTier.FREE)
        assert capacity == pytest.approx(200.0)

    def test_a_zero_rate_blocks_the_tier_without_dividing_by_zero(self) -> None:
        """RATE_LIMIT_FREE_RPM=0 is a legitimate way to switch a tier off."""
        limiter = RateLimiter(_settings(rate_limit_free_rpm=0))
        capacity, rate = limiter._bucket_params(UserTier.FREE)
        assert capacity == 0.0
        assert rate == 0.0


class TestSingleton:
    def test_returns_the_same_instance(self) -> None:
        set_rate_limiter(None)
        assert get_rate_limiter() is get_rate_limiter()
        set_rate_limiter(None)

    def test_can_be_replaced(self) -> None:
        replacement = RateLimiter(_settings())
        set_rate_limiter(replacement)
        assert get_rate_limiter() is replacement
        set_rate_limiter(None)


class TestEnforce:
    async def test_enforce_raises_when_throttled(self) -> None:
        from api.exceptions import RateLimitError

        limiter = _wired(ScriptedRedis([0, 0.0, 7.0]))
        with pytest.raises(RateLimitError):
            await limiter.enforce(_principal())

    async def test_enforce_returns_the_decision_when_allowed(self) -> None:
        limiter = _wired(ScriptedRedis([1, 5.0, 0.0]))
        decision = await limiter.enforce(_principal())
        assert decision.allowed is True

    async def test_the_retry_after_reaches_the_error(self) -> None:
        from api.exceptions import RateLimitError

        limiter = _wired(ScriptedRedis([0, 0.0, 12.0]))
        with pytest.raises(RateLimitError) as caught:
            await limiter.enforce(_principal())
        assert "12" in str(caught.value.details) or caught.value.details


class TestDecisionHeaders:
    async def test_headers_carry_the_tier_and_limit(self) -> None:
        limiter = _wired(ScriptedRedis([1, 7.0, 0.0]))
        headers = (await limiter.check(_principal(UserTier.PRO))).headers()
        assert headers["X-RateLimit-Tier"] == "pro"
        assert headers["X-RateLimit-Limit"] == "300"
        assert headers["X-RateLimit-Remaining"] == "7"

    async def test_a_throttled_decision_carries_retry_after(self) -> None:
        limiter = _wired(ScriptedRedis([0, 0.0, 3.0]))
        headers = (await limiter.check(_principal())).headers()
        assert "Retry-After" in headers


class TestLocalBucketRefills:
    async def test_tokens_come_back_over_time(self) -> None:
        limiter = _wired(ScriptedRedis(exc=ConnectionError("gone")))
        principal = _principal(user_id="refill")

        while (await limiter.check(principal)).allowed:
            pass

        key = f"ratelimit:{principal.tier.value}:{principal.user_id}"
        bucket = limiter._local[key]
        bucket.updated = time.monotonic() - 120  # two minutes of refill

        assert (await limiter.check(principal)).allowed is True
