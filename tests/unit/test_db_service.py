"""Unit tests for the inference log and batch-job store.

These run against a real in-memory SQLite database rather than mocks, so the
SQL is genuinely executed: the aggregate in `inference_stats`, the filtered
ordering in `recent_predictions`, and the commit/rollback behaviour of
`session()` are all things a mocked session would happily let through while
broken.

Two behaviours matter more than the rest and get the most attention here.

**Writes must never raise.** By the time `log_inference` runs, the caller's
prediction has already been computed and is on its way back to them. A logging
failure that propagates would turn a successful request into a 500 — trading
the user's result for a row in a table nobody is reading yet.

**Every method must work when the database is down.** The service is optional
infrastructure: the API is expected to keep serving predictions without it, so
each method has a defined answer for the unavailable case rather than an
exception.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import SQLAlchemyError

from api.config import Settings
from api.services.db_service import DatabaseService, get_db_service, set_db_service


def _settings(url: str = "sqlite+aiosqlite:///:memory:") -> Settings:
    return Settings(environment="test", database_url=url)


@pytest.fixture
async def db() -> DatabaseService:
    """A connected service backed by in-memory SQLite, with tables created."""
    service = DatabaseService(_settings())
    assert await service.connect(create_tables=True)
    yield service
    await service.close()


@pytest.fixture
def down() -> DatabaseService:
    """A service that was never connected."""
    return DatabaseService(_settings())


def _log(**overrides) -> dict:
    record = {
        "correlation_id": "corr-1",
        "user_id": "user-1",
        "user_tier": "pro",
        "task": "classification",
        "model_name": "resnet50",
        "model_version": "1.0.0",
        "runtime": "onnx",
        "device": "cpu",
        "image_hash": "a" * 64,
        "top_label": "tabby",
        "top_confidence": 0.91,
        "num_results": 5,
        "total_ms": 42.0,
        "inference_ms": 30.0,
        "success": True,
    }
    record.update(overrides)
    return record


def _job(job_id: str = "job-1", **overrides) -> dict:
    record = {
        "id": job_id,
        "correlation_id": "corr-1",
        "task": "classification",
        "status": "pending",
        "total_items": 4,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
class TestConnect:
    async def test_connects_to_sqlite(self, db: DatabaseService) -> None:
        assert db.available is True

    async def test_an_unreachable_database_returns_false_rather_than_raising(self) -> None:
        """Startup must survive a database that is not up yet."""
        service = DatabaseService(_settings("postgresql+asyncpg://u:p@127.0.0.1:5499/nothing-here"))
        assert await service.connect() is False
        assert service.available is False

    async def test_a_nonsense_url_is_reported_not_raised(self) -> None:
        service = DatabaseService(_settings("not-a-database-url"))
        assert await service.connect() is False

    async def test_the_failure_reason_is_kept(self) -> None:
        service = DatabaseService(_settings("not-a-database-url"))
        await service.connect()
        assert (await service.health())["error"]

    async def test_sqlite_skips_pool_configuration(self) -> None:
        """StaticPool raises TypeError if handed pool_size, so it must not be."""
        service = DatabaseService(
            Settings(
                environment="test",
                database_url="sqlite+aiosqlite:///:memory:",
                database_pool_size=17,
                database_max_overflow=5,
            )
        )
        assert await service.connect(create_tables=True) is True
        await service.close()

    async def test_tables_are_only_created_when_asked(self) -> None:
        service = DatabaseService(_settings())
        await service.connect(create_tables=False)
        # The table is absent, so the write fails - and is swallowed, as designed.
        assert await service.log_inference(_log()) is None
        await service.close()

    async def test_close_is_idempotent(self, db: DatabaseService) -> None:
        await db.close()
        await db.close()
        assert db.available is False

    async def test_close_without_connecting_is_safe(self, down: DatabaseService) -> None:
        await down.close()


class TestAvailability:
    async def test_unavailable_before_connect(self, down: DatabaseService) -> None:
        assert down.available is False

    async def test_ping_on_a_live_database(self, db: DatabaseService) -> None:
        assert await db.ping() is True

    async def test_ping_without_an_engine_is_false(self, down: DatabaseService) -> None:
        assert await down.ping() is False

    async def test_ping_after_close_is_false(self, db: DatabaseService) -> None:
        await db.close()
        assert await db.ping() is False

    async def test_ping_marks_the_service_unavailable_on_failure(self) -> None:
        """A database that dies mid-life must be noticed, not assumed healthy.

        The engine is swapped for a stub that refuses connections, rather than
        patching the live one: SQLAlchemy engines do not survive having their
        `connect` replaced and restored around a test.
        """

        class RefusingEngine:
            def connect(self, *args, **kwargs):
                raise SQLAlchemyError("server closed the connection")

            async def dispose(self) -> None:
                return None

        service = DatabaseService(_settings())
        assert await service.connect(create_tables=True)
        assert service.available is True

        service._engine = RefusingEngine()
        assert await service.ping() is False
        assert service.available is False
        assert (await service.health())["status"] == "unavailable"

        await service.close()


class TestHealth:
    async def test_healthy_reports_latency(self, db: DatabaseService) -> None:
        health = await db.health()
        assert health["status"] == "healthy"
        assert health["latency_ms"] >= 0
        assert health["error"] is None

    async def test_unavailable_reports_the_error(self, down: DatabaseService) -> None:
        health = await down.health()
        assert health["status"] == "unavailable"
        assert health["latency_ms"] is None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
class TestSession:
    async def test_commits_on_success(self, db: DatabaseService) -> None:
        from db.models import InferenceLog

        async with db.session() as session:
            session.add(InferenceLog(**_log()))
        stats = await db.inference_stats(hours=1)
        assert stats["models"][0]["total"] == 1

    async def test_rolls_back_and_re_raises_on_error(self, db: DatabaseService) -> None:
        """A half-written transaction is worse than no write at all."""
        from db.models import InferenceLog

        with pytest.raises(RuntimeError, match="deliberate"):
            async with db.session() as session:
                session.add(InferenceLog(**_log()))
                raise RuntimeError("deliberate")

        assert await db.inference_stats(hours=1) == {"window_hours": 1, "models": []}

    async def test_using_a_session_before_connect_is_a_clear_error(
        self, down: DatabaseService
    ) -> None:
        with pytest.raises(RuntimeError, match="connect"):
            async with down.session():
                pass


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
class TestLogInference:
    async def test_returns_the_new_row_id(self, db: DatabaseService) -> None:
        assert await db.log_inference(_log()) == 1

    async def test_ids_increment(self, db: DatabaseService) -> None:
        first = await db.log_inference(_log())
        second = await db.log_inference(_log(correlation_id="corr-2"))
        assert second > first

    async def test_returns_none_when_the_database_is_down(self, down: DatabaseService) -> None:
        assert await down.log_inference(_log()) is None

    async def test_a_bad_record_is_swallowed_not_raised(self, db: DatabaseService) -> None:
        """The prediction is already on its way to the caller."""
        assert await db.log_inference({"nonsense_column": 1}) is None

    async def test_a_failed_write_does_not_break_the_next_one(self, db: DatabaseService) -> None:
        await db.log_inference({"nonsense_column": 1})
        assert await db.log_inference(_log()) is not None

    async def test_a_failure_record_can_be_logged(self, db: DatabaseService) -> None:
        row = await db.log_inference(
            _log(
                success=False,
                error_code="MODEL_LOAD_ERROR",
                error_message="boom",
                top_label=None,
                top_confidence=None,
            )
        )
        assert row is not None


class TestBatchJobs:
    async def test_create_then_read(self, db: DatabaseService) -> None:
        assert await db.create_batch_job(_job()) is True
        job = await db.get_batch_job("job-1")
        assert job is not None
        assert job.status == "pending"
        assert job.total_items == 4

    async def test_reading_an_unknown_job_returns_none(self, db: DatabaseService) -> None:
        assert await db.get_batch_job("no-such-job") is None

    async def test_update_changes_the_row(self, db: DatabaseService) -> None:
        await db.create_batch_job(_job())
        assert await db.update_batch_job("job-1", status="completed", completed_items=4) is True
        job = await db.get_batch_job("job-1")
        assert job.status == "completed"
        assert job.completed_items == 4

    async def test_updating_an_unknown_job_reports_success_but_changes_nothing(
        self, db: DatabaseService
    ) -> None:
        """SQL UPDATE matching no rows is not an error."""
        assert await db.update_batch_job("ghost", status="completed") is True
        assert await db.get_batch_job("ghost") is None

    async def test_a_duplicate_id_is_reported_not_raised(self, db: DatabaseService) -> None:
        await db.create_batch_job(_job())
        assert await db.create_batch_job(_job()) is False

    async def test_an_unknown_column_is_reported_not_raised(self, db: DatabaseService) -> None:
        await db.create_batch_job(_job())
        assert await db.update_batch_job("job-1", not_a_column="x") is False

    async def test_a_read_failure_returns_none_rather_than_raising(
        self, db: DatabaseService, monkeypatch
    ) -> None:
        def explode(*a, **k):
            raise SQLAlchemyError("connection reset")

        monkeypatch.setattr(type(db), "session", explode)
        assert await db.get_batch_job("job-1") is None

    async def test_every_method_is_safe_when_the_database_is_down(
        self, down: DatabaseService
    ) -> None:
        assert await down.create_batch_job(_job()) is False
        assert await down.get_batch_job("job-1") is None
        assert await down.update_batch_job("job-1", status="x") is False


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
class TestRecentPredictions:
    async def test_returns_label_and_confidence(self, db: DatabaseService) -> None:
        await db.log_inference(_log())
        rows = await db.recent_predictions("resnet50")
        assert len(rows) == 1
        assert rows[0]["label"] == "tabby"
        assert rows[0]["confidence"] == pytest.approx(0.91)

    async def test_filters_by_model_name(self, db: DatabaseService) -> None:
        await db.log_inference(_log(model_name="resnet50"))
        await db.log_inference(_log(model_name="yolov8n", correlation_id="c2"))
        assert len(await db.recent_predictions("resnet50")) == 1

    async def test_filters_by_version_when_given(self, db: DatabaseService) -> None:
        await db.log_inference(_log(model_version="1.0.0"))
        await db.log_inference(_log(model_version="2.0.0", correlation_id="c2"))
        assert len(await db.recent_predictions("resnet50", "2.0.0")) == 1
        assert len(await db.recent_predictions("resnet50")) == 2

    async def test_excludes_failed_inferences(self, db: DatabaseService) -> None:
        """Drift analysis over failed requests would measure the outage."""
        await db.log_inference(_log(success=True))
        await db.log_inference(_log(success=False, correlation_id="c2"))
        assert len(await db.recent_predictions("resnet50")) == 1

    async def test_respects_the_since_window(self, db: DatabaseService) -> None:
        await db.log_inference(_log())
        future = datetime.now(UTC) + timedelta(hours=1)
        assert await db.recent_predictions("resnet50", since=future) == []

    async def test_defaults_to_the_last_seven_days(self, db: DatabaseService) -> None:
        await db.log_inference(_log())
        assert len(await db.recent_predictions("resnet50")) == 1

    async def test_honours_the_limit(self, db: DatabaseService) -> None:
        for i in range(5):
            await db.log_inference(_log(correlation_id=f"c{i}"))
        assert len(await db.recent_predictions("resnet50", limit=2)) == 2

    async def test_returns_empty_when_the_database_is_down(self, down: DatabaseService) -> None:
        assert await down.recent_predictions("resnet50") == []

    async def test_a_query_failure_returns_empty_rather_than_raising(
        self, db: DatabaseService, monkeypatch
    ) -> None:
        def explode(*a, **k):
            raise SQLAlchemyError("connection reset")

        monkeypatch.setattr(type(db), "session", explode)
        assert await db.recent_predictions("resnet50") == []


class TestInferenceStats:
    async def test_aggregates_by_model_and_version(self, db: DatabaseService) -> None:
        await db.log_inference(_log(model_name="resnet50", model_version="1.0.0"))
        await db.log_inference(
            _log(model_name="resnet50", model_version="1.0.0", correlation_id="c2")
        )
        await db.log_inference(
            _log(model_name="yolov8n", model_version="1.0.0", correlation_id="c3")
        )
        stats = await db.inference_stats(hours=1)
        by_model = {m["model"]: m for m in stats["models"]}
        assert by_model["resnet50:1.0.0"]["total"] == 2
        assert by_model["yolov8n:1.0.0"]["total"] == 1

    async def test_counts_successes_separately(self, db: DatabaseService) -> None:
        await db.log_inference(_log(success=True))
        await db.log_inference(_log(success=False, correlation_id="c2"))
        stats = await db.inference_stats(hours=1)
        row = stats["models"][0]
        assert row["total"] == 2
        assert row["successes"] == 1
        assert row["success_rate"] == pytest.approx(0.5)

    async def test_reports_latency_aggregates(self, db: DatabaseService) -> None:
        await db.log_inference(_log(total_ms=10.0))
        await db.log_inference(_log(total_ms=30.0, correlation_id="c2"))
        row = (await db.inference_stats(hours=1))["models"][0]
        assert row["avg_ms"] == pytest.approx(20.0)
        assert row["max_ms"] == pytest.approx(30.0)

    async def test_an_empty_window_is_not_an_error(self, db: DatabaseService) -> None:
        assert (await db.inference_stats(hours=1))["models"] == []

    async def test_the_window_is_honoured(self, db: DatabaseService) -> None:
        await db.log_inference(_log())
        assert (await db.inference_stats(hours=24))["window_hours"] == 24

    async def test_returns_empty_when_the_database_is_down(self, down: DatabaseService) -> None:
        assert await down.inference_stats() == {}

    async def test_a_query_failure_returns_empty_rather_than_raising(
        self, db: DatabaseService, monkeypatch
    ) -> None:
        def explode(*a, **k):
            raise SQLAlchemyError("connection reset")

        monkeypatch.setattr(type(db), "session", explode)
        assert await db.inference_stats() == {}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
class TestSingleton:
    def test_returns_the_same_instance(self) -> None:
        set_db_service(None)
        assert get_db_service() is get_db_service()
        set_db_service(None)

    def test_can_be_replaced(self) -> None:
        replacement = DatabaseService(_settings())
        set_db_service(replacement)
        assert get_db_service() is replacement
        set_db_service(None)

    def test_clearing_creates_a_fresh_one(self) -> None:
        first = get_db_service()
        set_db_service(None)
        assert get_db_service() is not first
        set_db_service(None)
