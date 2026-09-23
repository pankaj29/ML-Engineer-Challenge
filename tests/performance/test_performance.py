"""Performance tests: stress testing and memory profiling.

Plain English:
    Functional tests ask "is the answer right?". These ask "is it fast enough,
    and does it stay fast?" Two failure modes matter in production and neither
    shows up in a unit test:

    * **Latency under load.** A model that answers in 80 ms when idle can take
      3 seconds when twenty requests arrive together. The brief requires
      sub-second single-image inference, and that has to hold under
      concurrency, not just on an empty box.

    * **Memory leaks.** ML services leak slowly — a tensor kept alive by a
      logging reference, a cache without a bound. The symptom is a container
      that gets OOM-killed after six hours, which no functional test catches.
      Here we run many iterations and check that memory levels off.

These are marked ``performance`` and excluded from the default CI run, because
timing assertions on shared CI hardware are flaky. Run them deliberately::

    pytest tests/performance -m performance -v
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time

import pytest

pytestmark = [pytest.mark.performance, pytest.mark.slow]


def _rss_mb() -> float:
    """Resident memory of this process, in megabytes."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1_048_576
    except ImportError:
        pytest.skip("psutil is not installed")
        return 0.0


class TestSingleImageLatency:
    """The brief's headline requirement: sub-second single-image inference."""

    async def test_classification_latency(self, inference_service, sample_image: bytes) -> None:
        await inference_service.classify(sample_image)  # warm up

        timings = []
        for _ in range(20):
            start = time.perf_counter()
            await inference_service.classify(sample_image, use_cache=False)
            timings.append((time.perf_counter() - start) * 1000)

        timings.sort()
        p50 = timings[len(timings) // 2]
        p95 = timings[int(len(timings) * 0.95)]

        print(f"\nclassification: p50={p50:.1f} ms  p95={p95:.1f} ms")
        assert p50 < 1000, f"median {p50:.0f} ms exceeds the 1 s requirement"

    async def test_detection_latency(self, inference_service, sample_image: bytes) -> None:
        await inference_service.detect(sample_image)

        timings = []
        for _ in range(10):
            start = time.perf_counter()
            await inference_service.detect(sample_image, use_cache=False)
            timings.append((time.perf_counter() - start) * 1000)

        timings.sort()
        p50 = timings[len(timings) // 2]
        print(f"\ndetection: p50={p50:.1f} ms")
        assert p50 < 1000


class TestConcurrentLoad:
    """Behaviour when many requests arrive at once."""

    async def test_concurrent_requests_complete(
        self, inference_service, sample_image: bytes
    ) -> None:
        """Twenty simultaneous requests must all be answered or cleanly shed."""
        from api.exceptions import AppError

        async def one() -> object:
            try:
                return await inference_service.classify(sample_image, use_cache=False)
            except AppError as exc:
                return exc

        start = time.perf_counter()
        results = await asyncio.gather(*[one() for _ in range(20)])
        elapsed = time.perf_counter() - start

        succeeded = sum(1 for r in results if not isinstance(r, Exception))
        shed = sum(1 for r in results if isinstance(r, AppError))

        print(f"\n20 concurrent: {succeeded} ok, {shed} shed, {elapsed:.2f} s total")
        # Every request must reach a definite outcome — none may hang.
        assert succeeded + shed == 20
        # Shedding is acceptable under load; silent failure is not.
        assert succeeded > 0

    async def test_concurrency_limit_is_respected(
        self, fake_model_service, null_cache, sample_image: bytes
    ) -> None:
        """In-flight count must never exceed the configured ceiling."""
        from api.config import Settings
        from api.services.inference_service import InferenceService
        from tests.conftest import FakeRuntime

        fake_model_service.models["classification"].runtime = FakeRuntime(delay=0.05)
        service = InferenceService(
            fake_model_service,
            null_cache,
            Settings(environment="test", max_concurrent_inferences=4),
        )

        peak = 0

        async def watch() -> None:
            nonlocal peak
            for _ in range(200):
                peak = max(peak, service.inflight)
                await asyncio.sleep(0.005)

        watcher = asyncio.create_task(watch())

        async def call() -> None:
            try:
                await service.classify(sample_image, use_cache=False)
            except Exception:
                pass

        await asyncio.gather(*[call() for _ in range(20)])
        watcher.cancel()

        print(f"\npeak in-flight: {peak} (limit 4)")
        assert peak <= 4, f"concurrency limit breached: saw {peak} in flight"

    async def test_throughput(self, inference_service, sample_image: bytes) -> None:
        """Measure sustained requests per second."""
        await inference_service.classify(sample_image)

        count = 30
        start = time.perf_counter()
        await asyncio.gather(
            *[inference_service.classify(sample_image, use_cache=False) for _ in range(count)]
        )
        elapsed = time.perf_counter() - start

        rps = count / elapsed
        print(f"\nthroughput: {rps:.1f} req/s over {count} requests")
        assert rps > 1.0


class TestMemoryProfile:
    """Memory must level off, not climb."""

    async def test_repeated_inference_does_not_leak(
        self, inference_service, sample_image: bytes
    ) -> None:
        """Run many inferences and check memory stabilises.

        Memory is expected to *rise* at first — arenas, caches, interned
        objects. What matters is that the second half grows far less than the
        first. A genuine leak grows linearly and shows equal growth in both.
        """
        for _ in range(10):  # warm up allocators
            await inference_service.classify(sample_image, use_cache=False)
        gc.collect()

        baseline = _rss_mb()

        for _ in range(50):
            await inference_service.classify(sample_image, use_cache=False)
        gc.collect()
        midpoint = _rss_mb()

        for _ in range(50):
            await inference_service.classify(sample_image, use_cache=False)
        gc.collect()
        final = _rss_mb()

        first_half = midpoint - baseline
        second_half = final - midpoint

        print(
            f"\nmemory: baseline {baseline:.1f} MB -> {midpoint:.1f} MB -> {final:.1f} MB "
            f"(growth: {first_half:+.1f} then {second_half:+.1f} MB)"
        )

        # Total growth over 100 inferences must stay modest.
        assert final - baseline < 150, f"grew {final - baseline:.0f} MB over 100 inferences"
        # And growth must be decelerating, not linear.
        if first_half > 5:
            assert second_half < first_half * 1.5, "memory growth is not levelling off"

    async def test_batch_memory_is_released(self, inference_service, sample_image: bytes) -> None:
        """A large batch must not permanently retain its buffers."""
        gc.collect()
        baseline = _rss_mb()

        for _ in range(5):
            await asyncio.gather(
                *[inference_service.classify(sample_image, use_cache=False) for _ in range(16)]
            )
        gc.collect()
        after = _rss_mb()

        print(f"\nbatch memory: {baseline:.1f} -> {after:.1f} MB ({after - baseline:+.1f})")
        assert after - baseline < 200

    def test_preprocessing_does_not_leak(self, sample_image: bytes) -> None:
        """Preprocessing runs on every request; a leak here compounds fast."""
        from api.utils.image_processing import CLASSIFICATION_PREPROCESS, preprocess

        for _ in range(20):
            preprocess(sample_image, CLASSIFICATION_PREPROCESS)
        gc.collect()
        baseline = _rss_mb()

        for _ in range(500):
            preprocess(sample_image, CLASSIFICATION_PREPROCESS)
        gc.collect()
        after = _rss_mb()

        print(f"\npreprocessing x500: {after - baseline:+.1f} MB")
        assert after - baseline < 50


class TestStressInference:
    """Sustained load and pathological inputs."""

    async def test_sustained_load(self, inference_service, sample_image: bytes) -> None:
        """Latency must not degrade over a sustained run.

        A service that starts fast and slows down under steady load usually
        has a growing queue or an unbounded cache.
        """
        await inference_service.classify(sample_image)

        early, late = [], []
        for i in range(40):
            start = time.perf_counter()
            await inference_service.classify(sample_image, use_cache=False)
            elapsed = (time.perf_counter() - start) * 1000
            (early if i < 20 else late).append(elapsed)

        early_median = statistics.median(early)
        late_median = statistics.median(late)

        print(f"\nsustained: first half {early_median:.1f} ms, second half {late_median:.1f} ms")
        # Allow generous headroom for normal jitter on a shared machine.
        assert late_median < early_median * 3, "latency degraded badly under sustained load"

    @pytest.mark.parametrize(
        ("width", "height"),
        [(64, 64), (224, 224), (1920, 1080), (4000, 200), (200, 4000)],
    )
    async def test_handles_varied_image_shapes(
        self, inference_service, width: int, height: int
    ) -> None:
        """Extreme aspect ratios must not crash or blow the latency budget."""
        from tests.conftest import make_image

        image = make_image(width, height)
        start = time.perf_counter()
        response = await inference_service.classify(image, use_cache=False)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"\n{width}x{height}: {elapsed:.1f} ms")
        assert response.predictions
        assert elapsed < 2000

    async def test_error_path_is_not_slower_than_success(
        self, inference_service, not_an_image: bytes
    ) -> None:
        """Rejecting bad input must be fast — it is the cheap path.

        If validation were slower than inference, an attacker could exhaust
        the service with junk uploads more cheaply than with real ones.
        """
        from api.exceptions import AppError
        from api.utils.validators import validate_image_bytes

        start = time.perf_counter()
        for _ in range(100):
            try:
                validate_image_bytes(not_an_image)
            except AppError:
                pass
        elapsed = (time.perf_counter() - start) * 1000 / 100

        print(f"\nrejection cost: {elapsed:.3f} ms per invalid image")
        assert elapsed < 10, "rejecting invalid input is too expensive"
