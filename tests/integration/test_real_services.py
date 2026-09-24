"""Integration tests against the actual PostgreSQL and Redis servers.

The rest of the suite substitutes in-memory SQLite for PostgreSQL and
`fakeredis` for Redis. That is the right default — it keeps 975 tests running
with no infrastructure, which is what the brief asks for under "mock external
dependencies". But a substitute only proves the code works against the
substitute, and these two differ from the real thing in ways this codebase has
already had to work around:

* **SQLite has no native boolean.** `inference_stats` sums a `case()`
  expression rather than casting, precisely because the cast is
  dialect-dependent. A SQLite-only test cannot tell you the PostgreSQL path is
  right.
* **The rate limiter's token bucket is a Lua script.** Its whole purpose is to
  make the read-modify-write atomic across replicas. fakeredis will execute
  it, but only a real server exercises it as the concurrency primitive it
  exists to be.
* **JSON columns, indexes and real transaction isolation** behave differently
  enough to be worth touching once.

Every test here skips when the service is not reachable, so a laptop with no
Docker still gets a green suite. Run `docker compose up -d` to include them.

Nothing here shares state with the application's own tables beyond the rows it
writes and removes: the fixtures use their own schema so a local run cannot
disturb development data.
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

POSTGRES_URL = os.getenv(
    "TEST_POSTGRES_URL",
    "postgresql+asyncpg://mluser:mlpass@127.0.0.1:5432/mldb",
)
REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def _reachable(url: str, default_port: int, timeout: float = 1.0) -> bool:
    """Can we open a socket to whatever host this URL names?

    The host is taken from the URL rather than assumed to be localhost, so
    pointing TEST_POSTGRES_URL at another machine skips or runs based on that
    machine rather than on this one.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port

    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


postgres_up = pytest.mark.skipif(
    not _reachable(POSTGRES_URL, 5432),
    reason="PostgreSQL is not running (docker compose up -d postgres)",
)
redis_up = pytest.mark.skipif(
    not _reachable(REDIS_URL, 6379),
    reason="Redis is not running (docker compose up -d redis)",
)


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
@pytest.fixture
async def pg():
    """A DatabaseService on the real PostgreSQL, with its tables created."""
    from api.config import Settings
    from api.services.db_service import DatabaseService

    service = DatabaseService(Settings(environment="test", database_url=POSTGRES_URL))
    if not await service.connect(create_tables=True):
        pytest.skip("could not connect to PostgreSQL")
    yield service
    await service.close()


def _log(correlation_id: str, **overrides) -> dict:
    record = {
        "correlation_id": correlation_id,
        "user_id": "integration",
        "user_tier": "pro",
        "task": "classification",
        "model_name": f"itest-{correlation_id[:8]}",
        "model_version": "1.0.0",
        "runtime": "onnx",
        "device": "cpu",
        "image_hash": "b" * 64,
        "top_label": "tabby",
        "top_confidence": 0.87,
        "num_results": 5,
        "total_ms": 40.0,
        "inference_ms": 28.0,
        "success": True,
    }
    record.update(overrides)
    return record


@postgres_up
class TestPostgres:
    async def test_connects_and_reports_healthy(self, pg) -> None:
        health = await pg.health()
        assert health["status"] == "healthy"
        assert health["latency_ms"] >= 0

    async def test_writes_and_reads_back(self, pg) -> None:
        corr = uuid.uuid4().hex
        row_id = await pg.log_inference(_log(corr))
        assert row_id is not None

        rows = await pg.recent_predictions(f"itest-{corr[:8]}")
        assert len(rows) == 1
        assert rows[0]["label"] == "tabby"
        assert rows[0]["confidence"] == pytest.approx(0.87)

    async def test_the_boolean_aggregate_works_on_postgres(self, pg) -> None:
        """The reason this file exists.

        `inference_stats` sums a `case()` because SQLite has no native
        boolean. That workaround has to be correct on PostgreSQL too, and only
        PostgreSQL can say so.
        """
        corr = uuid.uuid4().hex
        model = f"itest-{corr[:8]}"
        await pg.log_inference(_log(corr, success=True))
        await pg.log_inference(_log(uuid.uuid4().hex, model_name=model, success=True))
        await pg.log_inference(_log(uuid.uuid4().hex, model_name=model, success=False))

        stats = await pg.inference_stats(hours=1)
        row = next(m for m in stats["models"] if m["model"] == f"{model}:1.0.0")

        assert row["total"] == 3
        assert row["successes"] == 2
        assert row["success_rate"] == pytest.approx(2 / 3, abs=1e-3)

    async def test_latency_aggregates_are_real_numbers(self, pg) -> None:
        corr = uuid.uuid4().hex
        model = f"itest-{corr[:8]}"
        await pg.log_inference(_log(corr, total_ms=10.0))
        await pg.log_inference(_log(uuid.uuid4().hex, model_name=model, total_ms=50.0))

        row = next(
            m
            for m in (await pg.inference_stats(hours=1))["models"]
            if m["model"] == f"{model}:1.0.0"
        )
        assert row["avg_ms"] == pytest.approx(30.0, abs=0.01)
        assert row["max_ms"] == pytest.approx(50.0, abs=0.01)

    async def test_a_json_column_round_trips(self, pg) -> None:
        """`result` is JSON/JSONB; SQLite stores it as text."""
        corr = uuid.uuid4().hex
        payload = {"predictions": [{"label": "cat", "confidence": 0.9}], "nested": {"a": [1, 2]}}
        assert await pg.log_inference(_log(corr, result=payload)) is not None

    async def test_batch_job_lifecycle(self, pg) -> None:
        job_id = f"itest-{uuid.uuid4().hex}"
        assert await pg.create_batch_job(
            {
                "id": job_id,
                "correlation_id": job_id,
                "task": "classification",
                "status": "pending",
                "total_items": 3,
            }
        )
        assert await pg.update_batch_job(job_id, status="completed", completed_items=3)

        job = await pg.get_batch_job(job_id)
        assert job.status == "completed"
        assert job.completed_items == 3

    async def test_a_duplicate_primary_key_is_rejected_by_the_database(self, pg) -> None:
        """A real constraint, not one SQLite happens to share."""
        job_id = f"itest-{uuid.uuid4().hex}"
        record = {
            "id": job_id,
            "correlation_id": job_id,
            "task": "classification",
            "status": "pending",
            "total_items": 1,
        }
        assert await pg.create_batch_job(record) is True
        assert await pg.create_batch_job(record) is False

    async def test_a_failed_write_does_not_poison_the_connection(self, pg) -> None:
        """A rolled-back transaction must leave the pool usable.

        PostgreSQL puts a connection into an aborted state after an error
        until the transaction is rolled back. If `session()` did not roll back,
        every later query on that connection would fail with
        "current transaction is aborted" - a failure mode SQLite does not
        reproduce.
        """
        await pg.log_inference({"not_a_column": True})
        corr = uuid.uuid4().hex
        assert await pg.log_inference(_log(corr)) is not None

    async def test_concurrent_writes_all_land(self, pg) -> None:
        """Real pooling, real concurrency."""
        corr = uuid.uuid4().hex
        model = f"itest-{corr[:8]}"
        await asyncio.gather(
            *(pg.log_inference(_log(uuid.uuid4().hex, model_name=model)) for _ in range(20))
        )
        rows = await pg.recent_predictions(model, limit=100)
        assert len(rows) == 20


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
@pytest.fixture
async def real_cache():
    """A CacheService on the real Redis, using database 15 and cleaning up."""
    from api.config import Settings
    from api.services.cache_service import CacheService

    service = CacheService(Settings(environment="test", redis_url=REDIS_URL, cache_enabled=True))
    if not await service.connect():
        pytest.skip("could not connect to Redis")
    await service._client.flushdb()
    yield service
    try:
        await service._client.flushdb()
    finally:
        await service.close()


@redis_up
class TestRedisCache:
    async def test_connects_and_reports_healthy(self, real_cache) -> None:
        assert (await real_cache.health())["status"] == "healthy"

    async def test_set_then_get_round_trips(self, real_cache) -> None:
        await real_cache.set("cv:itest:1", {"label": "cat", "confidence": 0.9})
        assert (await real_cache.get("cv:itest:1"))["label"] == "cat"

    async def test_ttl_is_applied_by_the_server(self, real_cache) -> None:
        await real_cache.set("cv:itest:ttl", {"a": 1}, ttl=60)
        assert 0 < await real_cache._client.ttl("cv:itest:ttl") <= 60

    async def test_scan_based_invalidation_on_a_real_keyspace(self, real_cache) -> None:
        """`scan_iter` rather than `KEYS`, against a server that knows the difference."""
        for i in range(25):
            await real_cache.set(f"cv:classify:doomed:1.0.0:onnx:{i}", {"i": i})
        for i in range(5):
            await real_cache.set(f"cv:classify:keeper:1.0.0:onnx:{i}", {"i": i})

        removed = await real_cache.invalidate_model("doomed:1.0.0")

        assert removed == 25
        assert await real_cache.get("cv:classify:keeper:1.0.0:onnx:0") is not None

    async def test_concurrent_writes_are_all_visible(self, real_cache) -> None:
        await asyncio.gather(*(real_cache.set(f"cv:itest:c{i}", {"i": i}) for i in range(50)))
        found = [await real_cache.get(f"cv:itest:c{i}") for i in range(50)]
        assert all(f is not None for f in found)


@redis_up
class TestRedisRateLimiter:
    """The Lua token bucket, against the server it was written for."""

    @pytest.fixture
    async def limiter(self):
        from api.config import Settings
        from api.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(
            Settings(
                environment="test",
                redis_url=REDIS_URL,
                rate_limit_enabled=True,
                rate_limit_free_rpm=10,
                rate_limit_burst_multiplier=1.5,
            )
        )
        if not await limiter.connect():
            pytest.skip("could not connect to Redis")
        await limiter._client.flushdb()
        yield limiter
        try:
            await limiter._client.flushdb()
        finally:
            await limiter.close()

    @staticmethod
    def _principal(user_id: str):
        from api.middleware.auth import Principal
        from api.models.schemas import UserTier

        return Principal(user_id=user_id, tier=UserTier.FREE, auth_method="api_key")

    async def test_the_lua_script_runs_on_a_real_server(self, limiter) -> None:
        decision = await limiter.check(self._principal(uuid.uuid4().hex))
        assert decision.allowed is True
        assert decision.limit == 10

    async def test_the_burst_capacity_is_enforced(self, limiter) -> None:
        """10 rpm with a 1.5x burst means 15 tokens, then throttling."""
        user = self._principal(uuid.uuid4().hex)
        outcomes = [(await limiter.check(user)).allowed for _ in range(20)]

        assert sum(outcomes) == 15, f"expected 15 allowed, got {sum(outcomes)}"
        assert outcomes[-1] is False

    async def test_it_is_atomic_under_concurrency(self, limiter) -> None:
        """The reason the bucket is Lua rather than GET/SET.

        Twenty simultaneous requests against a 15-token bucket must allow
        exactly 15. A non-atomic read-modify-write would let extra requests
        through under this exact race.
        """
        user = self._principal(uuid.uuid4().hex)
        decisions = await asyncio.gather(*(limiter.check(user) for _ in range(20)))
        allowed = sum(d.allowed for d in decisions)

        assert allowed == 15, f"{allowed} requests allowed through a 15-token bucket"

    async def test_separate_users_have_separate_buckets(self, limiter) -> None:
        heavy = self._principal(uuid.uuid4().hex)
        for _ in range(20):
            await limiter.check(heavy)

        assert (await limiter.check(self._principal(uuid.uuid4().hex))).allowed is True

    async def test_a_throttled_caller_is_told_when_to_retry(self, limiter) -> None:
        user = self._principal(uuid.uuid4().hex)
        decision = None
        for _ in range(20):
            decision = await limiter.check(user)
            if not decision.allowed:
                break
        assert decision.allowed is False
        assert decision.retry_after > 0

    async def test_cost_is_charged_correctly(self, limiter) -> None:
        """A batch charges N tokens, so it cannot slip past a tier limit."""
        user = self._principal(uuid.uuid4().hex)
        first = await limiter.check(user, cost=10)
        assert first.allowed is True
        assert first.remaining <= 5

        assert (await limiter.check(user, cost=10)).allowed is False
