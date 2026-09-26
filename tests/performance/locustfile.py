"""HTTP-level load testing with Locust.

Plain English:
    The tests in ``test_performance.py`` call the service's Python functions
    directly. This file goes through the real network: real HTTP requests,
    through Nginx, against a running stack. That catches everything the
    in-process tests cannot — connection limits, gateway rate limiting, proxy
    timeouts, and what actually happens to latency when 200 clients arrive at
    once.

Run it against a live stack::

    docker compose up -d
    locust -f tests/performance/locustfile.py --host http://localhost

    # ...or headless, writing a CSV report:
    locust -f tests/performance/locustfile.py --host http://localhost \\
           --users 50 --spawn-rate 5 --run-time 2m --headless \\
           --csv benchmarks/reports/loadtest

How to read the result:
    * **Failures** should be 0, or exclusively 429s if you are deliberately
      testing the rate limiter. A 5xx under load is a real defect.
    * **p95 latency** is the number that matters, not the average.
    * Watch p95 as users increase. The point where it starts climbing steeply
      is the service's actual capacity, and it is almost always lower than the
      point where errors begin.
"""

from __future__ import annotations

import base64
import io
import random

from locust import HttpUser, between, events, task

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # pragma: no cover - locust is an optional dev dependency
    raise SystemExit(
        "locust load tests need pillow and numpy: pip install -r requirements-dev.txt"
    ) from exc


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------
def _make_image(width: int, height: int, seed: int) -> bytes:
    """Build a random-noise JPEG.

    Noise rather than flat colour on purpose: a flat image compresses to
    almost nothing, so it would under-state both the network cost and the
    decode cost of real traffic.
    """
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# A spread of sizes, generated once at import so image creation is never part
# of the measured request time.
IMAGE_POOL: list[tuple[str, bytes]] = [
    ("small_224", _make_image(224, 224, 1)),
    ("medium_640", _make_image(640, 480, 2)),
    ("large_1920", _make_image(1920, 1080, 3)),
]
B64_POOL: list[tuple[str, str]] = [
    (name, base64.b64encode(data).decode()) for name, data in IMAGE_POOL
]

# A small pool of *repeated* images, so the cache is genuinely exercised.
# Using a unique image every time would measure a 0% cache hit rate, which is
# not what production looks like.
CACHED_IMAGE = B64_POOL[0][1]

API_KEY = "dev-key-pro"


class MLApiUser(HttpUser):
    """Simulates a typical API client.

    ``wait_time`` models think time between requests. Without it, every
    simulated user hammers as fast as it can, which measures maximum
    throughput rather than behaviour under realistic load.
    """

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        """Set the auth header once per simulated user.

        Only the API key is set session-wide. Content-Type is deliberately NOT
        pinned: `json=` sets it per request, and a sticky
        `Content-Type: application/json` overrides the multipart boundary
        header on file uploads, so the server correctly rejects them with 422.
        """
        self.client.headers.update({"X-API-Key": API_KEY})

    # --- The common case ---------------------------------------------------
    @task(50)
    def classify_cached(self) -> None:
        """Classify a repeated image. Should mostly hit the cache."""
        with self.client.post(
            "/api/v1/classify",
            json={"image_base64": CACHED_IMAGE, "top_k": 5},
            name="/classify (cache-friendly)",
            catch_response=True,
        ) as response:
            if response.status_code == 429:
                # Rate limiting is the system working correctly, not a failure.
                response.success()
            elif response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")

    @task(20)
    def classify_unique(self) -> None:
        """Classify a varied image. Always a cache miss, so it measures the
        true cost of inference."""
        name, image = random.choice(B64_POOL)
        with self.client.post(
            "/api/v1/classify",
            json={"image_base64": image, "top_k": 5, "image_id": f"{name}-{random.random()}"},
            name="/classify (cache miss)",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 429):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(15)
    def detect(self) -> None:
        """Object detection — the most expensive single-image call."""
        _, image = random.choice(B64_POOL)
        with self.client.post(
            "/api/v1/detect",
            json={"image_base64": image, "confidence_threshold": 0.25},
            name="/detect",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 429):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(8)
    def embed(self) -> None:
        """Similarity embedding."""
        _, image = random.choice(B64_POOL)
        with self.client.post(
            "/api/v1/similarity/embed",
            json={"image_base64": image},
            name="/similarity/embed",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 429):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(4)
    def upload_multipart(self) -> None:
        """The multipart path, which a browser would use."""
        name, data = random.choice(IMAGE_POOL)
        with self.client.post(
            "/api/v1/classify/upload",
            files={"file": (f"{name}.jpg", data, "image/jpeg")},
            data={"top_k": "3"},
            name="/classify/upload",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 429):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def list_models(self) -> None:
        """A cheap metadata call, for contrast with the expensive ones."""
        self.client.get("/api/v1/models", name="/models")

    @task(1)
    def health(self) -> None:
        """What a load balancer does, constantly."""
        self.client.get("/api/v1/health", name="/health")


class BatchUser(HttpUser):
    """Simulates a client submitting background batches.

    Separate from :class:`MLApiUser` because batch traffic has a completely
    different shape: rare, large, and asynchronous. Mixing the two into one
    user class would make both sets of numbers hard to interpret.

    Run only this class with::

        locust -f tests/performance/locustfile.py --host http://localhost BatchUser
    """

    wait_time = between(5, 15)

    def on_start(self) -> None:
        # Auth only; see the note in MLApiUser.on_start.
        self.client.headers.update({"X-API-Key": API_KEY})

    @task
    def submit_and_poll(self) -> None:
        """Submit a batch, then poll until it finishes."""
        # Small images only. Eight uncompressible 1920x1080 JPEGs, base64
        # encoded, exceed the gateway's 12 MB body limit and come back as 413
        # — which measures the limit, not the service.
        items = [{"image_base64": B64_POOL[0][1], "image_id": f"item-{i}"} for i in range(8)]
        with self.client.post(
            "/api/v1/batch",
            json={"task": "classification", "items": items, "top_k": 3},
            name="/batch (submit)",
            catch_response=True,
        ) as response:
            if response.status_code == 429:
                response.success()
                return
            if response.status_code != 202:
                response.failure(f"HTTP {response.status_code}")
                return
            job_id = response.json()["job_id"]

        # Poll a bounded number of times. An unbounded poll loop in a load
        # test is how you accidentally DDoS your own status endpoint.
        #
        # 429 is expected here: the gateway rate-limits /api/v1/batch harder
        # than the inference endpoints, and a polling client is exactly what
        # that limit exists to contain.
        for _ in range(10):
            with self.client.get(
                f"/api/v1/batch/{job_id}",
                name="/batch/{job_id} (poll)",
                catch_response=True,
            ) as response:
                if response.status_code in (200, 429):
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}")


class HeavyUser(HttpUser):
    """Worst-case client: large images, no think time, no cache hits.

    Use this to find the breaking point rather than to measure normal
    behaviour::

        locust -f tests/performance/locustfile.py --host http://localhost HeavyUser
    """

    wait_time = between(0, 0.1)

    def on_start(self) -> None:
        # Auth only; see the note in MLApiUser.on_start.
        self.client.headers.update({"X-API-Key": API_KEY})

    @task
    def large_detection(self) -> None:
        _, image = B64_POOL[-1]  # the 1920x1080 image
        with self.client.post(
            "/api/v1/detect",
            json={
                "image_base64": image,
                "confidence_threshold": 0.1,  # more boxes survive filtering
                "max_detections": 300,
                "image_id": str(random.random()),  # defeat the cache
            },
            name="/detect (large, uncached)",
            catch_response=True,
        ) as response:
            # 503 is the service shedding load on purpose, which is the
            # correct behaviour under overload — not a failure.
            if response.status_code in (200, 429, 503):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Print what is about to run, so a CSV report is self-describing."""
    print(f"\nLoad test starting against {environment.host}")
    print(f"Image pool: {[name for name, _ in IMAGE_POOL]}")
    print("Note: 429 responses are counted as successes — rate limiting is a feature.\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Summarise, and say plainly whether the run passed."""
    stats = environment.stats.total
    print("\n" + "=" * 68)
    print("LOAD TEST SUMMARY")
    print("=" * 68)
    print(f"  requests        : {stats.num_requests}")
    print(f"  failures        : {stats.num_failures}")
    print(f"  median (p50)    : {stats.median_response_time} ms")
    print(f"  p95             : {stats.get_response_time_percentile(0.95)} ms")
    print(f"  p99             : {stats.get_response_time_percentile(0.99)} ms")
    print(f"  throughput      : {stats.total_rps:.1f} req/s")

    failure_rate = stats.num_failures / stats.num_requests if stats.num_requests else 0
    p95 = stats.get_response_time_percentile(0.95) or 0

    print("\n  verdict:")
    if failure_rate > 0.01:
        print(f"    FAIL — {failure_rate:.1%} of requests failed (excluding rate limiting)")
    elif p95 > 2000:
        print(f"    WARN — p95 of {p95} ms is above the 2 s target")
    else:
        print(
            f"    PASS — failures {failure_rate:.2%} (under 1%) and p95 of {p95} ms "
            "within the 2 s target"
        )
    print("=" * 68 + "\n")
