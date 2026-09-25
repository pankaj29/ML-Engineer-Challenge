"""End-to-end tests against the deployed Docker stack.

Every request here goes over HTTP to `http://localhost`, so it crosses nginx,
the API container, Redis, PostgreSQL and the Celery worker exactly as a real
client would. That is the difference from `tests/integration/`, which drives
the app in-process through `TestClient` and therefore never touches the
gateway, the container image, the network, or the real worker.

The gap is not theoretical. A container once served a model from the previous
day while the host had a newer one: the bind mount had not propagated and the
image predated the change. Every unit and integration test passed, because
they import the code directly and read the artifact from the working tree.
Only a request through the gateway could see it, so
`TestTheContainerIsNotStale` checks it explicitly.

Everything skips when the stack is not running, so a laptop with Docker closed
still has a green suite:

    docker compose up -d
    pytest tests/e2e -v
"""

from __future__ import annotations

import base64
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

BASE = "http://localhost/api/v1"
PRO_KEY = "dev-key-pro"
FREE_KEY = "dev-key-free"
SAMPLES = Path("samples")


def _gateway_up(timeout: float = 2.0) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect(("127.0.0.1", 80))
        return True
    except OSError:
        return False
    finally:
        sock.close()


_GATEWAY_UP = _gateway_up()

# Skipping is a convenience for a laptop with Docker closed. Where the stack
# is genuinely expected, skipping is a hole: pytest exits 0 when everything
# skips, so a stack that failed to come up would produce silent skips and a
# green build.
#
# The switch is E2E_REQUIRED rather than CI. CI is set for every job, and the
# unit-test job has no stack - keying off it made collecting this file abort
# that job's entire session before a single test ran. Only the CI step that
# starts the stack sets E2E_REQUIRED.
if os.getenv("E2E_REQUIRED") and not _GATEWAY_UP:
    raise RuntimeError(
        "E2E_REQUIRED is set but nothing is listening on port 80. Bring the "
        "stack up with `docker compose up -d` before running this suite."
    )

stack_up = pytest.mark.skipif(
    not _GATEWAY_UP,
    reason="the stack is not running (docker compose up -d)",
)


def call(
    path: str,
    body: dict | None = None,
    *,
    method: str = "POST",
    key: str | None = PRO_KEY,
    timeout: int = 90,
) -> tuple[int, dict | str, dict]:
    """One HTTP call through the gateway. Returns (status, parsed body, headers)."""
    request = urllib.request.Request(f"{BASE}{path}", method=method)
    if key:
        request.add_header("X-API-Key", key)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, data, timeout=timeout) as response:
            raw = response.read().decode()
            headers = {k.lower(): v for k, v in response.headers.items()}
            try:
                return response.status, json.loads(raw), headers
            except json.JSONDecodeError:
                return response.status, raw, headers
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        headers = {k.lower(): v for k, v in exc.headers.items()}
        try:
            return exc.code, json.loads(raw), headers
        except json.JSONDecodeError:
            return exc.code, raw, headers


@pytest.fixture(scope="module")
def dog() -> str:
    path = SAMPLES / "dog.jpg"
    if not path.is_file():
        pytest.skip("samples/dog.jpg is missing")
    return base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------------
@stack_up
class TestTheStackIsServing:
    def test_health_through_the_gateway(self) -> None:
        status, body, _ = call("/health", method="GET", key=None)
        assert status in (200, 503)
        assert body["status"] in {"healthy", "degraded", "unhealthy"}

    def test_every_dependency_is_reachable_from_inside_the_container(self) -> None:
        """Redis and PostgreSQL are only truly wired up when the container can
        reach them by service name on the compose network."""
        _, body, _ = call("/health", method="GET", key=None)
        components = {c["name"]: c["status"] for c in body["components"]}
        assert components.get("cache") == "healthy", components
        assert components.get("database") == "healthy", components

    def test_liveness_and_readiness(self) -> None:
        assert call("/health/live", method="GET", key=None)[0] == 200
        assert call("/health/ready", method="GET", key=None)[0] in (200, 503)

    def test_the_gateway_adds_a_correlation_id(self) -> None:
        _, _, headers = call("/health/live", method="GET", key=None)
        assert headers.get("x-correlation-id")


@stack_up
class TestTheContainerIsNotStale:
    """The failure that motivated this file.

    A bind-mounted artifact replaced on the host does not always reach the
    container on Docker Desktop, and code baked into the image never does
    without a rebuild. When the model, the registry and the preprocessing are
    all stale *together* the stack looks perfectly healthy while serving the
    previous release.
    """

    def test_the_served_registry_matches_the_working_tree(self) -> None:
        from api.services.model_service import ModelService

        status, body, _ = call("/models", method="GET")
        assert status == 200
        served = {m["name"]: m for m in body["models"]}

        for entry in ModelService().list_entries():
            assert entry.name in served, f"{entry.name} is registered but not served"
            assert served[entry.name]["input_shape"] == list(entry.input_shape), (
                f"{entry.name}: container serves input_shape "
                f"{served[entry.name]['input_shape']}, working tree says "
                f"{entry.input_shape}. The container is running older content."
            )

    def test_the_served_model_agrees_with_the_local_artifact(self, dog: str) -> None:
        """Run the same image through the container and through the local ONNX.

        Different predictions mean the two are different files.
        """
        import numpy as np
        import onnxruntime as ort

        from api.utils.image_processing import CLASSIFICATION_PREPROCESS, preprocess

        status, body, _ = call("/classify", {"image_base64": dog, "top_k": 1})
        assert status == 200
        served_label = body["predictions"][0]["label"]

        artifact = Path("models/artifacts/resnet50.onnx")
        if not artifact.is_file() or artifact.stat().st_size < 10_000:
            pytest.skip("local resnet50.onnx is absent or an LFS pointer")

        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        array = preprocess(base64.b64decode(dog), CLASSIFICATION_PREPROCESS).array
        logits = session.run(None, {session.get_inputs()[0].name: array})[0][0]

        labels = json.loads(
            Path("models/artifacts/imagenet_labels.json").read_text(encoding="utf-8")
        )
        local_label = labels[int(np.argmax(logits))]

        assert served_label == local_label, (
            f"the container predicts {served_label!r} and the local artifact "
            f"predicts {local_label!r}. The container is serving a different file. "
            f"Try: docker compose up -d --build"
        )


@stack_up
class TestInferenceEndToEnd:
    def test_classify(self, dog: str) -> None:
        status, body, _ = call("/classify", {"image_base64": dog, "top_k": 5})
        assert status == 200
        assert len(body["predictions"]) == 5
        assert body["model"]["name"]
        assert body["timing"]["total_ms"] > 0

    def test_classify_matches_the_documented_sample(self, dog: str) -> None:
        _, body, _ = call("/classify", {"image_base64": dog, "top_k": 5})
        labels = [p["label"] for p in body["predictions"]]
        assert "Labrador retriever" in labels, f"README says otherwise: {labels}"

    def test_detect(self, dog: str) -> None:
        status, body, _ = call("/detect", {"image_base64": dog, "confidence_threshold": 0.25})
        assert status == 200
        assert "dog" in [d["label"] for d in body["detections"]]

    def test_similarity_round_trip(self, dog: str) -> None:
        indexed = call(
            "/similarity/index", {"image_base64": dog, "image_id": "e2e-dog", "label": "dog"}
        )
        assert indexed[0] in (200, 201), indexed[1]

        status, body, _ = call("/similarity/search", {"image_base64": dog, "top_k": 5})
        assert status == 200
        assert "e2e-dog" in [r["id"] for r in body["results"]]

    def test_an_int8_runtime_can_be_selected(self, dog: str) -> None:
        """Registered, advertised by /models, and actually servable."""
        _, models, _ = call("/models", method="GET")
        has_int8 = any("onnx_int8" in m["available_runtimes"] for m in models["models"])
        if not has_int8:
            pytest.skip("no INT8 artifact registered")

        status, body, _ = call(
            "/classify", {"image_base64": dog, "top_k": 3, "runtime": "onnx_int8"}
        )
        assert status == 200
        assert body["model"]["runtime"] == "onnx_int8"

    def test_multipart_upload(self) -> None:
        """The form the README recommends, because base64 in argv overflows."""
        import mimetypes
        import uuid

        path = SAMPLES / "dog.jpg"
        if not path.is_file():
            pytest.skip("samples/dog.jpg is missing")

        boundary = uuid.uuid4().hex
        content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts = [
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="top_k"\r\n\r\n5\r\n'
            f"--{boundary}--\r\n".encode(),
        ]
        payload = b"".join(parts)

        request = urllib.request.Request(f"{BASE}/classify/upload", method="POST")
        request.add_header("X-API-Key", PRO_KEY)
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(request, payload, timeout=90) as response:
            body = json.loads(response.read().decode())
        assert len(body["predictions"]) == 5


@stack_up
class TestBatchThroughTheRealWorker:
    """Integration tests cannot cover this: they run with Redis unreachable, so
    there is no broker and no worker. Only the deployed stack has both."""

    def test_a_batch_job_completes(self, dog: str) -> None:
        status, submitted, _ = call(
            "/batch",
            {
                "task": "classification",
                "top_k": 3,
                "items": [
                    {"image_base64": dog, "image_id": "a"},
                    {"image_base64": dog, "image_id": "b"},
                ],
            },
        )
        assert status in (200, 202), submitted
        job_id = submitted["job_id"]

        state = None
        for _ in range(45):
            time.sleep(2)
            _, state, _ = call(f"/batch/{job_id}", method="GET")
            if state["status"] in ("completed", "failed", "cancelled"):
                break

        assert state["status"] == "completed", state
        assert state["total_items"] == 2
        assert state["completed_items"] == 2
        assert state["failed_items"] == 0

    def test_an_unknown_job_is_404(self) -> None:
        status, _, _ = call("/batch/does-not-exist", method="GET")
        assert status == 404


@stack_up
class TestCachingAcrossTheNetwork:
    def test_a_repeated_request_is_served_from_redis(self, dog: str) -> None:
        body = {"image_base64": dog, "top_k": 3}
        call("/classify", body)
        status, second, _ = call("/classify", body)
        assert status == 200
        assert second["cached"] is True

    def test_a_cached_response_is_identical_to_the_original(self, dog: str) -> None:
        """Deliberately not a timing assertion.

        At 15-25 ms per request, network jitter is larger than the saving, so
        "warm is faster" fails at random. Timing lives in the performance
        suite, which CI excludes for this reason. What can be checked reliably
        is that the cached answer is the *same* answer, which is the property
        that would actually be wrong if the cache were broken.
        """
        body = {"image_base64": dog, "top_k": 4}
        _, first, _ = call("/classify", body)
        _, second, _ = call("/classify", body)

        assert second["cached"] is True
        assert first["predictions"] == second["predictions"]
        assert first["model"] == second["model"]

    def test_different_parameters_are_cached_separately(self, dog: str) -> None:
        """One cache entry serving every top_k would be a silent correctness bug."""
        _, three, _ = call("/classify", {"image_base64": dog, "top_k": 3})
        _, seven, _ = call("/classify", {"image_base64": dog, "top_k": 7})
        assert len(three["predictions"]) == 3
        assert len(seven["predictions"]) == 7


@stack_up
class TestSecurityAtTheEdge:
    def test_no_key_is_rejected(self, dog: str) -> None:
        assert call("/classify", {"image_base64": dog}, key=None)[0] == 401

    def test_a_bad_key_is_rejected(self, dog: str) -> None:
        assert call("/classify", {"image_base64": dog}, key="not-a-key")[0] == 401

    def test_a_non_image_is_rejected(self) -> None:
        payload = base64.b64encode(b"#!/bin/sh\nrm -rf /").decode()
        assert call("/classify", {"image_base64": payload})[0] in (400, 415, 422)

    def test_the_error_envelope_is_consistent(self) -> None:
        status, body, _ = call("/classify", {"image_base64": "not-base64-at-all"})
        assert status in (400, 415, 422)
        assert set(body["error"]) >= {"code", "message", "correlation_id"}

    def test_an_unknown_route_uses_the_same_envelope(self) -> None:
        status, body, _ = call("/there-is-no-such-endpoint", method="GET")
        assert status == 404
        assert "error" in body


@stack_up
class TestRateLimitingAcrossReplicas:
    def test_the_free_tier_is_throttled(self, dog: str) -> None:
        """Enforced by the Redis Lua bucket, shared by every replica."""
        codes = [call("/classify", {"image_base64": dog}, key=FREE_KEY)[0] for _ in range(20)]
        assert 429 in codes, f"free tier was never throttled: {codes}"

    def test_a_throttled_response_says_when_to_retry(self, dog: str) -> None:
        for _ in range(25):
            status, _, headers = call("/classify", {"image_base64": dog}, key=FREE_KEY)
            if status == 429:
                assert headers.get("retry-after") or headers.get("x-ratelimit-limit")
                return
        pytest.skip("the free tier did not throttle within 25 requests")

    def test_rate_limit_headers_are_present(self, dog: str) -> None:
        _, _, headers = call("/classify", {"image_base64": dog, "top_k": 1})
        assert headers.get("x-ratelimit-limit")
        assert headers.get("x-ratelimit-tier")


@stack_up
class TestObservability:
    def test_prometheus_metrics_are_exposed(self) -> None:
        status, body, _ = call("/metrics", method="GET", key=None)
        assert status == 200
        assert "inference" in str(body)

    def test_metrics_record_real_traffic(self, dog: str) -> None:
        """A metric defined but never incremented leaves a blank dashboard."""
        call("/classify", {"image_base64": dog, "top_k": 2})
        _, body, _ = call("/metrics", method="GET", key=None)
        text = str(body)

        for metric in ("inference_total", "http_requests_total", "inference_duration_seconds"):
            assert metric in text, f"{metric} is not exposed"

        # Defined-but-never-incremented is the failure mode: the dashboard
        # panel renders, permanently empty, and nobody notices.
        counted = [
            line
            for line in text.splitlines()
            if line.startswith("inference_total{") and not line.rstrip().endswith(" 0.0")
        ]
        assert counted, "inference_total is exposed but still zero after a real request"

    def test_a_supplied_correlation_id_is_echoed(self) -> None:
        request = urllib.request.Request(f"{BASE}/health/live", method="GET")
        request.add_header("X-Correlation-ID", "e2e-trace-42")
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.headers.get("x-correlation-id") == "e2e-trace-42"
