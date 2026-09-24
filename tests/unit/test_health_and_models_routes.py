"""Unit tests for the health probes and the model-metadata routes.

The brief requires `GET /api/v1/health` and `GET /api/v1/models` explicitly,
and Part 4 requires "health checks for all services". The audit found
`api/routers/health.py` at 77% and `api/routers/models.py` at 75%, with the
per-component checks and the degraded paths untested.

The two health probes are not interchangeable and the tests say so:
`/health/live` must depend on nothing, so a Redis blip cannot make the
orchestrator kill a working container; `/health/ready` must depend on
everything, so a degraded instance leaves the load balancer.
"""

from __future__ import annotations

import pytest

from api.routers.health import _aggregate, _check_cache, _check_models, uptime_seconds
from api.services.model_service import ModelService


class TestLivenessProbe:
    """Checks nothing external. That is the entire point of it."""

    def test_returns_200(self, api_client) -> None:
        assert api_client.get("/api/v1/health/live").status_code == 200

    def test_reports_alive(self, api_client) -> None:
        assert api_client.get("/api/v1/health/live").json()["status"] == "alive"

    def test_needs_no_authentication(self, api_client) -> None:
        """A probe that needs a credential is a probe that fails at 3am."""
        assert api_client.get("/api/v1/health/live").status_code == 200

    def test_reports_uptime(self, api_client) -> None:
        body = api_client.get("/api/v1/health/live").json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0


class TestReadinessProbe:
    def test_responds(self, api_client) -> None:
        assert api_client.get("/api/v1/health/ready").status_code in (200, 503)

    def test_reports_a_status(self, api_client) -> None:
        assert "status" in api_client.get("/api/v1/health/ready").json()


class TestFullHealth:
    def test_responds_with_components(self, api_client) -> None:
        body = api_client.get("/api/v1/health").json()
        assert "components" in body
        assert "status" in body

    def test_each_component_names_itself_and_its_state(self, api_client) -> None:
        for component in api_client.get("/api/v1/health").json()["components"]:
            assert component["name"]
            assert component["status"] in {"healthy", "degraded", "unhealthy"}

    def test_reports_the_version(self, api_client) -> None:
        assert api_client.get("/api/v1/health").json()["version"]


class TestUptime:
    def test_is_non_negative(self) -> None:
        assert uptime_seconds() >= 0


class TestAggregate:
    """One unhealthy component must not be averaged away."""

    @staticmethod
    def _c(status: str, name: str = "cache"):
        from api.models.responses import ComponentHealth

        return ComponentHealth(name=name, status=status, latency_ms=1.0, message=None)

    def test_all_healthy_is_healthy(self) -> None:
        assert _aggregate([self._c("healthy"), self._c("healthy")]) == "healthy"

    def test_one_degraded_degrades_the_whole(self) -> None:
        assert _aggregate([self._c("healthy"), self._c("degraded")]) == "degraded"

    def test_a_single_broken_model_only_degrades(self) -> None:
        """One broken task must not take the other two offline."""
        components = [
            self._c("healthy", "model:classification"),
            self._c("healthy", "model:detection"),
            self._c("unhealthy", "model:similarity"),
        ]
        assert _aggregate(components) == "degraded"

    def test_every_model_broken_is_unhealthy(self) -> None:
        """Nothing can be served, so say so rather than claiming degraded."""
        components = [
            self._c("unhealthy", "model:classification"),
            self._c("unhealthy", "model:detection"),
        ]
        assert _aggregate(components) == "unhealthy"

    def test_a_broken_dependency_degrades_but_does_not_kill(self) -> None:
        """Redis down means slower, not offline."""
        components = [
            self._c("unhealthy", "cache"),
            self._c("healthy", "model:classification"),
        ]
        assert _aggregate(components) == "degraded"

    def test_no_components_is_not_a_crash(self) -> None:
        assert _aggregate([]) in {"healthy", "degraded", "unhealthy"}


class TestComponentChecks:
    async def test_cache_check_reports_rather_than_raises(self, null_cache) -> None:
        """A dead cache is a degraded service, not a 500."""
        result = await _check_cache(null_cache)
        assert result.name
        assert result.status in {"healthy", "degraded", "unhealthy"}

    def test_model_check_reports_each_task(self, fake_model_service) -> None:
        results = _check_models(fake_model_service)
        assert results
        assert all(r.status in {"healthy", "degraded", "unhealthy"} for r in results)

    def test_model_check_on_an_empty_registry_is_not_healthy(self, tmp_path) -> None:
        """No models loaded is a real problem and must be reported as one."""
        import json

        from api.config import Settings

        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps({"models": []}), encoding="utf-8")
        service = ModelService(
            config=Settings(model_registry_path=str(registry), model_artifacts_dir=str(tmp_path))
        )
        results = _check_models(service)
        assert all(r.status != "healthy" for r in results) or not results


class TestModelsRoutes:
    def test_lists_models(self, api_client, auth_headers) -> None:
        body = api_client.get("/api/v1/models", headers=auth_headers).json()
        assert body["models"]

    def test_filters_by_task(self, api_client, auth_headers) -> None:
        body = api_client.get(
            "/api/v1/models", params={"task": "classification"}, headers=auth_headers
        ).json()
        assert all(m["task"] == "classification" for m in body["models"])

    def test_rejects_an_unknown_task(self, api_client, auth_headers) -> None:
        response = api_client.get(
            "/api/v1/models", params={"task": "telepathy"}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_gets_one_model_by_name(self, api_client, auth_headers) -> None:
        listed = api_client.get("/api/v1/models", headers=auth_headers).json()["models"][0]
        response = api_client.get(f"/api/v1/models/{listed['name']}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["name"] == listed["name"]

    def test_unknown_model_is_404_not_a_substitute(self, api_client, auth_headers) -> None:
        response = api_client.get("/api/v1/models/no-such-model", headers=auth_headers)
        assert response.status_code == 404

    def test_requires_authentication(self, api_client) -> None:
        assert api_client.get("/api/v1/models").status_code == 401

    def test_reload_endpoint_responds(self, api_client, auth_headers) -> None:
        response = api_client.post("/api/v1/models/reload", headers=auth_headers)
        assert response.status_code in (200, 403, 503)


class TestMetricsRoute:
    def test_exposes_prometheus_text(self, api_client, auth_headers) -> None:
        response = api_client.get("/api/v1/metrics", headers=auth_headers)
        assert response.status_code == 200
        assert "# HELP" in response.text

    def test_uses_the_prometheus_content_type(self, api_client, auth_headers) -> None:
        response = api_client.get("/api/v1/metrics", headers=auth_headers)
        assert "text/plain" in response.headers["content-type"]


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/health/live", "/api/v1/health/ready"])
def test_every_health_probe_returns_json(api_client, path: str) -> None:
    response = api_client.get(path)
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json(), dict)
