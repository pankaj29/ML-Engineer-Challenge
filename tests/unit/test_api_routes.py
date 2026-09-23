"""Unit tests for the API route handlers.

These drive the app through a TestClient with fake models underneath, so they
cover the full HTTP path — routing, validation, serialisation, error shaping —
without needing Redis, Postgres or real weights.
"""

from __future__ import annotations

import base64

import pytest


class TestRootAndDocs:
    def test_root_lists_endpoints(self, api_client) -> None:
        response = api_client.get("/")
        assert response.status_code == 200
        assert "endpoints" in response.json()

    def test_openapi_spec_is_generated(self, api_client) -> None:
        spec = api_client.get("/openapi.json").json()
        assert "/api/v1/classify" in spec["paths"]
        assert "ApiKeyAuth" in spec["components"]["securitySchemes"]

    def test_docs_render(self, api_client) -> None:
        assert api_client.get("/docs").status_code == 200


class TestAuthentication:
    def test_rejects_missing_credentials(self, api_client, sample_image_b64: str) -> None:
        response = api_client.post("/api/v1/classify", json={"image_base64": sample_image_b64})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"

    def test_rejects_wrong_key(self, api_client, sample_image_b64: str) -> None:
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64},
            headers={"X-API-Key": "definitely-not-valid"},
        )
        assert response.status_code == 401

    def test_accepts_valid_key(self, api_client, auth_headers, sample_image_b64: str) -> None:
        response = api_client.post(
            "/api/v1/classify", json={"image_base64": sample_image_b64}, headers=auth_headers
        )
        assert response.status_code == 200

    def test_accepts_jwt(self, api_client, sample_image_b64: str) -> None:
        from api.middleware.auth import create_access_token
        from api.models.schemas import UserTier

        token = create_access_token("user-1", UserTier.PRO)
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_rejects_expired_jwt(self, api_client, sample_image_b64: str) -> None:
        import jwt

        from api.config import settings

        expired = jwt.encode(
            {"sub": "u", "tier": "pro", "exp": 1_000_000_000},
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64},
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["details"]["reason"] == "expired"

    @pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/health/live", "/api/v1/metrics"])
    def test_public_paths_need_no_credentials(self, api_client, path: str) -> None:
        assert api_client.get(path).status_code in (200, 503)


class TestClassifyEndpoint:
    def test_returns_predictions(self, api_client, auth_headers, sample_image_b64: str) -> None:
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64, "top_k": 3},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 3
        assert body["model"]["task"] == "classification"
        assert body["cached"] is False

    def test_returns_correlation_id_header(
        self, api_client, auth_headers, sample_image_b64: str
    ) -> None:
        response = api_client.post(
            "/api/v1/classify", json={"image_base64": sample_image_b64}, headers=auth_headers
        )
        assert response.headers["x-correlation-id"]
        assert response.json()["correlation_id"] == response.headers["x-correlation-id"]

    def test_echoes_supplied_correlation_id(
        self, api_client, auth_headers, sample_image_b64: str
    ) -> None:
        """A trace id from an upstream service must be preserved, not replaced."""
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64},
            headers={**auth_headers, "X-Correlation-ID": "upstream-trace-123"},
        )
        assert response.headers["x-correlation-id"] == "upstream-trace-123"

    def test_multipart_upload(self, api_client, auth_headers, sample_image: bytes) -> None:
        response = api_client.post(
            "/api/v1/classify/upload",
            files={"file": ("cat.png", sample_image, "image/png")},
            data={"top_k": "2"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 2
        assert body["image_id"] == "cat.png"

    def test_accepts_data_uri_prefix(self, api_client, auth_headers, sample_image: bytes) -> None:
        encoded = base64.b64encode(sample_image).decode()
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": f"data:image/png;base64,{encoded}"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_echoes_image_id(self, api_client, auth_headers, sample_image_b64: str) -> None:
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64, "image_id": "photo-42"},
            headers=auth_headers,
        )
        assert response.json()["image_id"] == "photo-42"

    @pytest.mark.parametrize(
        ("payload", "expected_status"),
        [
            ({}, 422),  # no image at all
            ({"image_base64": ""}, 422),  # empty string
            ({"image_base64": "!!!!"}, 400),  # invalid base64
            ({"image_url": "file:///etc/passwd"}, 422),  # bad scheme
            ({"image_url": "http://169.254.169.254/"}, 422),  # SSRF
        ],
    )
    def test_invalid_payloads(
        self, api_client, auth_headers, payload: dict, expected_status: int
    ) -> None:
        response = api_client.post("/api/v1/classify", json=payload, headers=auth_headers)
        assert response.status_code == expected_status

    def test_rejects_both_image_sources(
        self, api_client, auth_headers, sample_image_b64: str
    ) -> None:
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64, "image_url": "https://x.com/a.png"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("top_k", [0, -1, 101, 9999])
    def test_rejects_out_of_range_top_k(
        self, api_client, auth_headers, sample_image_b64: str, top_k: int
    ) -> None:
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64, "top_k": top_k},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_rejects_unknown_field(self, api_client, auth_headers, sample_image_b64: str) -> None:
        """extra='forbid' catches typos instead of silently ignoring them."""
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64, "topk": 5},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_rejects_garbage_image(self, api_client, auth_headers, not_an_image: bytes) -> None:
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": base64.b64encode(not_an_image).decode()},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_IMAGE"

    def test_unknown_model_returns_404(
        self, api_client, auth_headers, sample_image_b64: str
    ) -> None:
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": sample_image_b64, "model_name": "nope"},
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


class TestDetectEndpoint:
    def test_returns_detections(self, api_client, auth_headers, sample_image_b64: str) -> None:
        response = api_client.post(
            "/api/v1/detect",
            json={"image_base64": sample_image_b64, "confidence_threshold": 0.5},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == len(body["detections"])
        assert body["image_width"] == 224

    def test_multipart_upload(self, api_client, auth_headers, sample_image: bytes) -> None:
        response = api_client.post(
            "/api/v1/detect/upload",
            files={"file": ("street.png", sample_image, "image/png")},
            headers=auth_headers,
        )
        assert response.status_code == 200

    @pytest.mark.parametrize("field", ["confidence_threshold", "iou_threshold"])
    def test_rejects_out_of_range_thresholds(
        self, api_client, auth_headers, sample_image_b64: str, field: str
    ) -> None:
        response = api_client.post(
            "/api/v1/detect",
            json={"image_base64": sample_image_b64, field: 1.5},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestSimilarityEndpoints:
    def test_embed(self, api_client, auth_headers, sample_image_b64: str) -> None:
        response = api_client.post(
            "/api/v1/similarity/embed",
            json={"image_base64": sample_image_b64},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["dimension"] == 2048

    def test_index_then_search_finds_itself(
        self, api_client, auth_headers, sample_image_b64: str
    ) -> None:
        """Indexing an image and searching for it must return it first, at ~1.0."""
        from api.services.similarity_index import SimilarityIndex, set_similarity_index

        set_similarity_index(SimilarityIndex(dimension=2048))

        indexed = api_client.post(
            "/api/v1/similarity/index",
            json={"image_base64": sample_image_b64, "label": "the original"},
            headers=auth_headers,
        )
        assert indexed.status_code == 201

        found = api_client.post(
            "/api/v1/similarity/search",
            json={"image_base64": sample_image_b64, "top_k": 5},
            headers=auth_headers,
        )
        assert found.status_code == 200
        body = found.json()
        assert body["count"] == 1
        assert body["results"][0]["label"] == "the original"
        assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-4)

        set_similarity_index(None)

    def test_search_on_empty_index(self, api_client, auth_headers, sample_image_b64: str) -> None:
        """An empty index returns no hits, not an error."""
        from api.services.similarity_index import SimilarityIndex, set_similarity_index

        set_similarity_index(SimilarityIndex(dimension=2048))
        response = api_client.post(
            "/api/v1/similarity/search",
            json={"image_base64": sample_image_b64},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 0
        assert body["results"] == []
        assert body["index_size"] == 0
        set_similarity_index(None)


class TestModelsEndpoint:
    def test_lists_models(self, api_client, auth_headers) -> None:
        response = api_client.get("/api/v1/models", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        assert set(body["defaults"]) == {"classification", "detection", "similarity"}

    def test_filters_by_task(self, api_client, auth_headers) -> None:
        response = api_client.get("/api/v1/models?task=classification", headers=auth_headers)
        assert response.json()["count"] == 1

    def test_rejects_invalid_task(self, api_client, auth_headers) -> None:
        assert (
            api_client.get("/api/v1/models?task=teleportation", headers=auth_headers).status_code
            == 422
        )

    def test_get_single_model(self, api_client, auth_headers) -> None:
        response = api_client.get("/api/v1/models/test-classifier", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["name"] == "test-classifier"

    def test_unknown_model_404(self, api_client, auth_headers) -> None:
        response = api_client.get("/api/v1/models/nonexistent", headers=auth_headers)
        assert response.status_code == 404


class TestHealthEndpoints:
    def test_health_returns_components(self, api_client) -> None:
        response = api_client.get("/api/v1/health")
        assert response.status_code in (200, 503)
        body = response.json()
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert {c["name"] for c in body["components"]} >= {"cache", "database"}

    def test_liveness_is_always_ok(self, api_client) -> None:
        """Liveness must not depend on any external service."""
        response = api_client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness(self, api_client) -> None:
        response = api_client.get("/api/v1/health/ready")
        assert response.status_code in (200, 503)
        assert "models_loaded" in response.json()


class TestMetricsEndpoint:
    def test_returns_prometheus_format(self, api_client) -> None:
        response = api_client.get("/api/v1/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "http_requests_total" in response.text

    def test_records_requests(self, api_client, auth_headers, sample_image_b64: str) -> None:
        api_client.post(
            "/api/v1/classify", json={"image_base64": sample_image_b64}, headers=auth_headers
        )
        assert "inference_total" in api_client.get("/api/v1/metrics").text


class TestErrorEnvelope:
    """Every error, from any source, must have the same shape."""

    @pytest.mark.parametrize(
        ("method", "path", "payload", "headers_key"),
        [
            ("get", "/api/v1/nonexistent", None, "auth"),
            ("post", "/api/v1/classify", {}, "auth"),
            ("post", "/api/v1/classify", {"image_base64": "x"}, "none"),
        ],
    )
    def test_error_shape_is_consistent(
        self, api_client, auth_headers, method: str, path: str, payload, headers_key: str
    ) -> None:
        headers = auth_headers if headers_key == "auth" else {}
        response = (
            api_client.get(path, headers=headers)
            if method == "get"
            else api_client.post(path, json=payload, headers=headers)
        )
        assert response.status_code >= 400
        body = response.json()
        assert "error" in body
        assert set(body["error"]) >= {"code", "message", "details", "timestamp"}
        assert isinstance(body["error"]["code"], str)

    def test_internal_details_are_not_leaked(
        self, api_client, auth_headers, not_an_image: bytes
    ) -> None:
        """A user-facing message must not contain file paths or stack traces."""
        response = api_client.post(
            "/api/v1/classify",
            json={"image_base64": base64.b64encode(not_an_image).decode()},
            headers=auth_headers,
        )
        message = response.json()["error"]["message"]
        assert "Traceback" not in message
        assert "/api/" not in message
        assert "C:\\" not in message
