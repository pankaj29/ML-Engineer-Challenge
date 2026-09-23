"""Integration tests for database operations.

These run against a real SQLAlchemy engine (SQLite in memory), so they
exercise actual SQL, real constraints and real transaction behaviour — not
mocks. The only thing swapped out is the database *engine*, which is why they
still run without a PostgreSQL server.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytestmark = pytest.mark.integration


class TestConnection:
    async def test_connects_and_pings(self, test_db) -> None:
        assert test_db.available is True
        assert await test_db.ping() is True

    async def test_health_reports_healthy(self, test_db) -> None:
        health = await test_db.health()
        assert health["status"] == "healthy"
        assert health["latency_ms"] is not None

    async def test_unreachable_database_fails_soft(self) -> None:
        """A bad connection string must not raise at startup, only report False."""
        from api.config import Settings
        from api.services.db_service import DatabaseService

        service = DatabaseService(
            Settings(
                environment="test",
                database_url="postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none",
            )
        )
        assert await service.connect() is False
        assert service.available is False
        await service.close()


class TestInferenceLog:
    def _record(self, **overrides) -> dict:
        record = {
            "correlation_id": "abc123",
            "user_id": "key_test",
            "user_tier": "pro",
            "task": "classification",
            "model_name": "resnet50",
            "model_version": "1.0.0",
            "runtime": "onnx",
            "device": "cpu",
            "image_hash": "deadbeef" * 8,
            "image_width": 224,
            "image_height": 224,
            "image_format": "PNG",
            "top_label": "goldfish",
            "top_confidence": 0.93,
            "num_results": 5,
            "total_ms": 42.1,
            "inference_ms": 31.0,
            "success": True,
        }
        record.update(overrides)
        return record

    async def test_writes_and_returns_id(self, test_db) -> None:
        row_id = await test_db.log_inference(self._record())
        assert isinstance(row_id, int)

    async def test_write_failure_returns_none_not_raise(self, test_db) -> None:
        """A malformed record must never break the request that produced it."""
        assert await test_db.log_inference({"not_a_real_column": 1}) is None

    async def test_writes_when_unavailable_are_noops(self) -> None:
        from api.services.db_service import DatabaseService

        service = DatabaseService()
        assert await service.log_inference(self._record()) is None

    async def test_recent_predictions_filters_by_model(self, test_db) -> None:
        for i in range(5):
            await test_db.log_inference(self._record(top_label=f"label_{i}"))
        await test_db.log_inference(self._record(model_name="other-model"))

        rows = await test_db.recent_predictions("resnet50")
        assert len(rows) == 5
        assert all(r["label"].startswith("label_") for r in rows)

    async def test_recent_predictions_filters_by_version(self, test_db) -> None:
        await test_db.log_inference(self._record(model_version="1.0.0"))
        await test_db.log_inference(self._record(model_version="2.0.0"))

        assert len(await test_db.recent_predictions("resnet50", "2.0.0")) == 1

    async def test_recent_predictions_excludes_failures(self, test_db) -> None:
        """Drift analysis must not be polluted by failed requests."""
        await test_db.log_inference(self._record(success=True))
        await test_db.log_inference(
            self._record(success=False, error_code="INFERENCE_FAILED", top_label=None)
        )
        assert len(await test_db.recent_predictions("resnet50")) == 1

    async def test_recent_predictions_respects_since(self, test_db) -> None:
        await test_db.log_inference(self._record())
        future = datetime.now(UTC) + timedelta(hours=1)
        assert await test_db.recent_predictions("resnet50", since=future) == []

    async def test_stores_error_details(self, test_db) -> None:
        row_id = await test_db.log_inference(
            self._record(
                success=False,
                error_code="INFERENCE_TIMEOUT",
                error_message="model exceeded its deadline",
                top_label=None,
                top_confidence=None,
            )
        )
        assert row_id is not None


class TestBatchJobs:
    def _job(self, job_id: str = "job-1", **overrides) -> dict:
        job = {
            "id": job_id,
            "correlation_id": "corr-1",
            "user_id": "key_test",
            "user_tier": "pro",
            "task": "classification",
            "model_name": "resnet50",
            "model_version": "1.0.0",
            "status": "pending",
            "total_items": 10,
            "submitted_at": datetime.now(UTC),
        }
        job.update(overrides)
        return job

    async def test_create_and_read(self, test_db) -> None:
        assert await test_db.create_batch_job(self._job()) is True
        job = await test_db.get_batch_job("job-1")
        assert job is not None
        assert job.total_items == 10
        assert job.status == "pending"

    async def test_missing_job_returns_none(self, test_db) -> None:
        assert await test_db.get_batch_job("does-not-exist") is None

    async def test_update_status_and_progress(self, test_db) -> None:
        await test_db.create_batch_job(self._job())
        await test_db.update_batch_job("job-1", status="running", completed_items=4, failed_items=1)

        job = await test_db.get_batch_job("job-1")
        assert job.status == "running"
        assert job.completed_items == 4
        assert job.progress_percent == 50.0

    async def test_progress_with_zero_items(self, test_db) -> None:
        await test_db.create_batch_job(self._job("job-empty", total_items=0))
        assert (await test_db.get_batch_job("job-empty")).progress_percent == 0.0

    async def test_duration_is_none_until_finished(self, test_db) -> None:
        await test_db.create_batch_job(self._job("job-2"))
        assert (await test_db.get_batch_job("job-2")).duration_seconds is None

    async def test_duration_after_completion(self, test_db) -> None:
        start = datetime.now(UTC)
        await test_db.create_batch_job(self._job("job-3"))
        await test_db.update_batch_job(
            "job-3",
            status="completed",
            started_at=start,
            completed_at=start + timedelta(seconds=42),
        )
        job = await test_db.get_batch_job("job-3")
        assert job.duration_seconds == pytest.approx(42.0, abs=1.0)

    async def test_duplicate_id_is_rejected(self, test_db) -> None:
        """A primary-key collision must fail cleanly rather than corrupt state."""
        assert await test_db.create_batch_job(self._job("dup")) is True
        assert await test_db.create_batch_job(self._job("dup")) is False

    async def test_stores_json_params(self, test_db) -> None:
        params = {"top_k": 5, "confidence_threshold": 0.25, "nested": {"a": [1, 2, 3]}}
        await test_db.create_batch_job(self._job("job-json", params=params))
        assert (await test_db.get_batch_job("job-json")).params == params


class TestStats:
    async def test_stats_over_window(self, test_db) -> None:
        for _ in range(3):
            await test_db.log_inference(
                {
                    "correlation_id": "c",
                    "task": "classification",
                    "model_name": "resnet50",
                    "model_version": "1.0.0",
                    "runtime": "onnx",
                    "image_hash": "x" * 64,
                    "total_ms": 50.0,
                    "success": True,
                }
            )
        stats = await test_db.inference_stats(hours=24)
        assert stats["window_hours"] == 24
        assert any(m["model"] == "resnet50:1.0.0" for m in stats["models"])
