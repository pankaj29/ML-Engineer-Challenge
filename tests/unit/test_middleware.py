"""Unit tests for authentication, rate limiting and monitoring middleware."""

from __future__ import annotations

import asyncio

import pytest

from api.config import Settings
from api.exceptions import AuthenticationError, AuthorizationError, RateLimitError
from api.middleware.auth import (
    Principal,
    create_access_token,
    is_public_path,
    verify_api_key,
    verify_jwt,
)
from api.middleware.monitoring import render_metrics
from api.middleware.rate_limit import RateLimiter
from api.models.schemas import UserTier


@pytest.fixture
def auth_settings() -> Settings:
    return Settings(
        environment="test",
        api_keys="key-free:free,key-basic:basic,key-pro:pro,key-ent:enterprise",
        jwt_secret="a-test-secret-that-is-definitely-long-enough-32chars",
        auth_enabled=True,
    )


class TestApiKeyAuth:
    @pytest.mark.parametrize(
        ("key", "tier"),
        [
            ("key-free", "free"),
            ("key-basic", "basic"),
            ("key-pro", "pro"),
            ("key-ent", "enterprise"),
        ],
    )
    def test_maps_key_to_tier(self, auth_settings: Settings, key: str, tier: str) -> None:
        principal = verify_api_key(key, auth_settings)
        assert principal.tier.value == tier
        assert principal.auth_method == "api_key"

    def test_rejects_unknown_key(self, auth_settings: Settings) -> None:
        with pytest.raises(AuthenticationError):
            verify_api_key("not-a-real-key", auth_settings)

    def test_user_id_does_not_contain_the_key(self, auth_settings: Settings) -> None:
        """The key must never appear in a user id that lands in logs."""
        principal = verify_api_key("key-pro", auth_settings)
        assert "key-pro" not in principal.user_id

    def test_same_key_gives_stable_user_id(self, auth_settings: Settings) -> None:
        assert (
            verify_api_key("key-pro", auth_settings).user_id
            == verify_api_key("key-pro", auth_settings).user_id
        )

    def test_no_keys_configured_is_an_error(self) -> None:
        """Auth enabled with no keys must fail closed, not open."""
        with pytest.raises(AuthenticationError):
            verify_api_key("anything", Settings(environment="test", api_keys="", auth_enabled=True))

    def test_malformed_config_entries_are_skipped(self) -> None:
        config = Settings(environment="test", api_keys="good:pro,,malformed,also-bad:")
        assert config.parsed_api_keys() == {"good": "pro"}


class TestJwtAuth:
    def test_round_trip(self, auth_settings: Settings) -> None:
        token = create_access_token("user-99", UserTier.PRO, ["classify"], auth_settings)
        principal = verify_jwt(token, auth_settings)
        assert principal.user_id == "user-99"
        assert principal.tier == UserTier.PRO
        assert principal.scopes == ("classify",)

    def test_rejects_tampered_token(self, auth_settings: Settings) -> None:
        token = create_access_token("user-99", UserTier.FREE, config=auth_settings)
        head, payload, signature = token.split(".")
        with pytest.raises(AuthenticationError):
            verify_jwt(f"{head}.{payload}.{signature[:-4]}AAAA", auth_settings)

    def test_rejects_token_signed_with_another_secret(self, auth_settings: Settings) -> None:
        other = Settings(
            environment="test", jwt_secret="a-completely-different-secret-key-32-chars"
        )
        token = create_access_token("attacker", UserTier.ENTERPRISE, config=other)
        with pytest.raises(AuthenticationError):
            verify_jwt(token, auth_settings)

    def test_rejects_alg_none_token(self, auth_settings: Settings) -> None:
        """The classic JWT forgery: declare 'alg: none' and omit the signature."""
        import base64
        import json

        def b64(data: dict) -> str:
            raw = json.dumps(data).encode()
            return base64.urlsafe_b64encode(raw).decode().rstrip("=")

        forged = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64({'sub': 'attacker', 'tier': 'enterprise', 'exp': 9999999999})}."
        with pytest.raises(AuthenticationError):
            verify_jwt(forged, auth_settings)

    def test_rejects_garbage(self, auth_settings: Settings) -> None:
        with pytest.raises(AuthenticationError):
            verify_jwt("not.a.jwt", auth_settings)

    def test_unknown_tier_falls_back_to_free(self, auth_settings: Settings) -> None:
        import jwt

        token = jwt.encode(
            {"sub": "u", "tier": "diamond", "exp": 9999999999},
            auth_settings.jwt_secret,
            algorithm=auth_settings.jwt_algorithm,
        )
        assert verify_jwt(token, auth_settings).tier == UserTier.FREE


class TestScopes:
    def test_api_key_bypasses_scope_checks(self) -> None:
        Principal("u", UserTier.PRO, "api_key").require_scope("admin")  # must not raise

    def test_jwt_without_scope_is_denied(self) -> None:
        with pytest.raises(AuthorizationError):
            Principal("u", UserTier.PRO, "jwt", ("classify",)).require_scope("admin")

    def test_jwt_with_scope_is_allowed(self) -> None:
        Principal("u", UserTier.PRO, "jwt", ("admin",)).require_scope("admin")


class TestPublicPaths:
    @pytest.mark.parametrize(
        "path",
        ["/", "/docs", "/openapi.json", "/api/v1/health", "/api/v1/health/live", "/api/v1/metrics"],
    )
    def test_public(self, path: str) -> None:
        assert is_public_path(path) is True

    @pytest.mark.parametrize("path", ["/api/v1/classify", "/api/v1/models", "/api/v1/batch"])
    def test_protected(self, path: str) -> None:
        assert is_public_path(path) is False


class TestRateLimiter:
    """The local-bucket fallback path; the Redis path is covered in integration.

    Each test passes ``rate_limit_enabled=True`` explicitly, because the suite
    sets ``RATE_LIMIT_ENABLED=false`` in the environment so that unrelated
    route tests are not throttled — and Settings reads environment variables.
    """

    async def test_allows_within_limit(self) -> None:
        limiter = RateLimiter(
            Settings(
                environment="test",
                rate_limit_enabled=True,
                rate_limit_free_rpm=10,
                rate_limit_burst_multiplier=1.0,
            )
        )
        principal = Principal("u1", UserTier.FREE, "api_key")
        for _ in range(10):
            assert (await limiter.check(principal)).allowed

    async def test_blocks_past_limit(self) -> None:
        limiter = RateLimiter(
            Settings(
                environment="test",
                rate_limit_enabled=True,
                rate_limit_free_rpm=3,
                rate_limit_burst_multiplier=1.0,
            )
        )
        principal = Principal("u2", UserTier.FREE, "api_key")
        outcomes = [(await limiter.check(principal)).allowed for _ in range(5)]
        assert outcomes == [True, True, True, False, False]

    async def test_tiers_have_separate_budgets(self) -> None:
        """One user exhausting their quota must not affect another user."""
        limiter = RateLimiter(
            Settings(
                environment="test",
                rate_limit_enabled=True,
                rate_limit_free_rpm=2,
                rate_limit_burst_multiplier=1.0,
            )
        )
        free = Principal("free-user", UserTier.FREE, "api_key")
        pro = Principal("pro-user", UserTier.PRO, "api_key")

        for _ in range(3):
            await limiter.check(free)
        assert (await limiter.check(free)).allowed is False
        assert (await limiter.check(pro)).allowed is True

    async def test_cost_consumes_multiple_tokens(self) -> None:
        """A 10-image batch must cost 10 tokens, not 1."""
        limiter = RateLimiter(
            Settings(
                environment="test",
                rate_limit_enabled=True,
                rate_limit_free_rpm=10,
                rate_limit_burst_multiplier=1.0,
            )
        )
        principal = Principal("u3", UserTier.FREE, "api_key")
        assert (await limiter.check(principal, cost=10)).allowed is True
        assert (await limiter.check(principal, cost=1)).allowed is False

    async def test_enforce_raises_with_retry_after(self) -> None:
        limiter = RateLimiter(
            Settings(
                environment="test",
                rate_limit_enabled=True,
                rate_limit_free_rpm=1,
                rate_limit_burst_multiplier=1.0,
            )
        )
        principal = Principal("u4", UserTier.FREE, "api_key")
        await limiter.enforce(principal)
        with pytest.raises(RateLimitError) as exc:
            await limiter.enforce(principal)
        assert exc.value.details["retry_after_seconds"] >= 1

    async def test_tokens_refill_over_time(self) -> None:
        """The bucket must refill continuously, not in a hard reset."""
        limiter = RateLimiter(
            Settings(
                environment="test",
                rate_limit_enabled=True,
                rate_limit_free_rpm=600,
                rate_limit_burst_multiplier=1.0,
            )
        )
        principal = Principal("u5", UserTier.FREE, "api_key")
        # 600 rpm = 10 tokens/second. Drain the bucket, then wait.
        for _ in range(600):
            await limiter.check(principal)
        assert (await limiter.check(principal)).allowed is False

        await asyncio.sleep(0.3)  # should refill ~3 tokens
        assert (await limiter.check(principal)).allowed is True

    async def test_disabled_limiter_always_allows(self) -> None:
        limiter = RateLimiter(Settings(environment="test", rate_limit_enabled=False))
        principal = Principal("u6", UserTier.FREE, "api_key")
        for _ in range(100):
            assert (await limiter.check(principal)).allowed

    async def test_headers_are_well_formed(self) -> None:
        limiter = RateLimiter(
            Settings(environment="test", rate_limit_enabled=True, rate_limit_pro_rpm=300)
        )
        decision = await limiter.check(Principal("u7", UserTier.PRO, "api_key"))
        headers = decision.headers()
        assert headers["X-RateLimit-Limit"] == "300"
        assert int(headers["X-RateLimit-Remaining"]) >= 0
        assert headers["X-RateLimit-Tier"] == "pro"


class TestMetrics:
    def test_renders_prometheus_text(self) -> None:
        output = render_metrics().decode()
        assert "# HELP" in output
        assert "# TYPE" in output

    def test_records_inference(self) -> None:
        from api.middleware.monitoring import record_inference

        record_inference(
            task="classification",
            model="m",
            version="1.0.0",
            runtime="onnx",
            duration_seconds=0.05,
        )
        assert 'task="classification"' in render_metrics().decode()

    def test_route_template_not_concrete_path(self) -> None:
        """Metric labels must use the route pattern to avoid cardinality blowup."""
        from api.middleware.monitoring import route_template

        class FakeRequest:
            scope = {"route": type("R", (), {"path": "/api/v1/batch/{job_id}"})()}

        assert route_template(FakeRequest()) == "/api/v1/batch/{job_id}"  # type: ignore[arg-type]

    def test_unmatched_route_is_bucketed(self) -> None:
        from api.middleware.monitoring import route_template

        class FakeRequest:
            scope: dict = {}

        assert route_template(FakeRequest()) == "unmatched"  # type: ignore[arg-type]


class TestLuaScriptIntegrity:
    """Guards the Redis token-bucket script against corruption.

    These exist because of a real bug: a `# noqa` comment was once placed
    directly after the opening triple-quote, which made it the first LINE OF
    THE LUA SCRIPT. Lua comments start with `--`, not `#`, so Redis would have
    rejected the script - but only once connected to a real Redis, which the
    unit tests never are (they exercise the local-bucket fallback). The bug
    was invisible to the whole suite.
    """

    def test_script_contains_no_python_comments(self) -> None:
        from api.middleware.rate_limit import _TOKEN_BUCKET_LUA

        assert "#" not in _TOKEN_BUCKET_LUA, (
            "A '#' in the Lua source means a Python comment leaked into the "
            "string. Redis will reject the script."
        )
        assert "noqa" not in _TOKEN_BUCKET_LUA

    def test_script_starts_with_lua(self) -> None:
        from api.middleware.rate_limit import _TOKEN_BUCKET_LUA

        assert _TOKEN_BUCKET_LUA.strip().splitlines()[0].startswith("local ")

    def test_script_declares_the_expected_contract(self) -> None:
        """The script must read the 4 ARGV values and return 3 results."""
        from api.middleware.rate_limit import _TOKEN_BUCKET_LUA

        for argv in ("ARGV[1]", "ARGV[2]", "ARGV[3]", "ARGV[4]"):
            assert argv in _TOKEN_BUCKET_LUA
        assert "KEYS[1]" in _TOKEN_BUCKET_LUA
        assert "return {allowed, tostring(tokens), tostring(retry_after)}" in _TOKEN_BUCKET_LUA

    def test_script_expires_idle_buckets(self) -> None:
        """Without EXPIRE, every caller who ever hit the API keeps a key forever."""
        from api.middleware.rate_limit import _TOKEN_BUCKET_LUA

        assert "EXPIRE" in _TOKEN_BUCKET_LUA
