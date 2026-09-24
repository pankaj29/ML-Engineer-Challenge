"""Unit tests for application startup, shutdown and the assembled app.

`api/main.py` sat at 64% because the lifespan handler - everything that
happens between "process starts" and "first request served" - was never
executed by a test. `create_app(testing=True)` deliberately skips it, which is
right for route tests and meant nobody ever ran the real thing.

The behaviour that matters here is **fail-soft startup**: the brief asks for
"graceful degradation", and the design commitment is that an unreachable Redis
or Postgres must not stop the API from serving predictions. These tests run
the real lifespan with nothing else running and assert exactly that.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.middleware.auth import AuthMiddleware
from api.middleware.monitoring import MonitoringMiddleware
from api.middleware.rate_limit import RateLimitMiddleware


@pytest.fixture
def degraded_app(monkeypatch):
    """The real app, with the real lifespan, and no infrastructure at all.

    Redis and Postgres are pointed at closed ports, so startup must take the
    degraded path rather than raising. Model warmup is disabled to keep this
    fast - it is covered separately in test_model_service_internals.py.

    The services default to the module-level `settings` object, which is built
    once when `api.config` is imported. Setting environment variables and
    clearing the `get_settings` cache does not reach it, so each service's
    module attribute is replaced directly. Without this the test quietly
    connected to whatever was really listening on the default ports and
    reported healthy - it passed only while no Redis happened to be running.
    """
    import api.services.cache_service as cache_service
    import api.services.db_service as db_service
    from api.config import Settings, get_settings

    monkeypatch.setenv("EAGER_MODEL_LOAD", "false")
    unreachable = Settings(
        redis_url="redis://127.0.0.1:6391/0",
        database_url="postgresql+asyncpg://u:p@127.0.0.1:5433/none",
        eager_model_load=False,
    )
    monkeypatch.setattr(cache_service, "settings", unreachable)
    monkeypatch.setattr(db_service, "settings", unreachable)

    get_settings.cache_clear()
    yield create_app()
    get_settings.cache_clear()


class TestStartupIsFailSoft:
    """An unreachable dependency must not stop the service from starting."""

    def test_starts_with_no_redis_and_no_database(self, degraded_app) -> None:
        with TestClient(degraded_app) as client:
            assert client.get("/api/v1/health/live").status_code == 200

    def test_reports_itself_degraded_rather_than_pretending(self, degraded_app) -> None:
        with TestClient(degraded_app) as client:
            body = client.get("/api/v1/health").json()
            assert body["status"] in {"degraded", "unhealthy"}

    def test_shuts_down_cleanly(self, degraded_app) -> None:
        """Exiting the context runs the shutdown half of the lifespan."""
        with TestClient(degraded_app) as client:
            client.get("/api/v1/health/live")
        # No exception on exit is the assertion.

    def test_services_are_wired_onto_app_state(self, degraded_app) -> None:
        with TestClient(degraded_app):
            assert degraded_app.state.inference_service is not None


class TestAppAssembly:
    """The app object itself, without running the lifespan."""

    def test_middleware_stack_is_in_the_documented_order(self) -> None:
        """Monitoring outermost, rate limiting innermost.

        Starlette applies the LAST added as the OUTERMOST layer, so the
        registration list reads inside-out. Getting this wrong is silent:
        auth after rate limiting would mean limiting an unidentified caller.
        """
        app = create_app(testing=True)
        classes = [m.cls for m in app.user_middleware]
        # Starlette stores this list outermost-first, so a request meets them
        # in exactly this order: index 0 sees it first.
        assert classes.index(MonitoringMiddleware) < classes.index(AuthMiddleware), (
            "Monitoring must be outermost so even rejected requests are timed "
            "and carry a correlation id"
        )
        assert classes.index(AuthMiddleware) < classes.index(RateLimitMiddleware), (
            "Auth must run before rate limiting - the limit depends on the "
            "caller's tier, which is unknown until they are identified"
        )

    def test_every_required_endpoint_is_mounted(self) -> None:
        """The six the brief names explicitly."""
        app = create_app(testing=True)
        paths = {getattr(r, "path", "") for r in app.routes}
        for required in (
            "/api/v1/classify",
            "/api/v1/detect",
            "/api/v1/batch",
            "/api/v1/models",
            "/api/v1/health",
            "/api/v1/metrics",
        ):
            assert required in paths, f"{required} is not mounted"

    def test_openapi_schema_generates(self) -> None:
        """A broken response model only shows up when the schema is built."""
        schema = create_app(testing=True).openapi()
        assert schema["info"]["title"]
        assert schema["paths"]

    def test_openapi_is_stable_across_calls(self) -> None:
        app = create_app(testing=True)
        assert app.openapi() == app.openapi()

    def test_docs_endpoints_are_exposed(self) -> None:
        app = create_app(testing=True)
        paths = {getattr(r, "path", "") for r in app.routes}
        assert "/docs" in paths
        assert "/openapi.json" in paths

    def test_root_returns_a_service_banner(self) -> None:
        with TestClient(create_app(testing=True)) as client:
            body = client.get("/").json()
            assert body


class TestErrorEnvelopeIsRegistered:
    def test_unknown_route_uses_the_standard_envelope(self, api_client) -> None:
        body = api_client.get("/api/v1/does-not-exist").json()
        assert "error" in body
        assert set(body["error"]) >= {"code", "message", "correlation_id"}

    def test_correlation_id_header_is_always_returned(self, api_client) -> None:
        response = api_client.get("/api/v1/health/live")
        assert response.headers.get("x-correlation-id")

    def test_supplied_correlation_id_is_echoed(self, api_client) -> None:
        """A caller's trace id must survive, so their logs join ours."""
        response = api_client.get(
            "/api/v1/health/live", headers={"X-Correlation-ID": "caller-trace-1"}
        )
        assert response.headers["x-correlation-id"] == "caller-trace-1"
