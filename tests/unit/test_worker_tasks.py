"""Unit tests for the Celery batch-processing task.

The task is invoked directly (``process_batch.run(...)``) rather than through
a broker, so these need no Redis. Services are replaced with fakes.

The central guarantee under test: **one bad image must not fail the batch.**
"""

from __future__ import annotations

import base64

import pytest


@pytest.fixture
def worker_task(fake_model_service, null_cache, monkeypatch):
    """Bind the real task to fake services."""
    from api.services.inference_service import InferenceService
    from worker.tasks import process_batch

    services = {
        "models": fake_model_service,
        "cache": null_cache,
        "inference": InferenceService(fake_model_service, null_cache),
    }
    # The task caches its services on the instance; pre-populate that cache.
    monkeypatch.setattr(type(process_batch), "_services", services, raising=False)

    class FakeState:
        def __init__(self) -> None:
            self.updates: list[dict] = []

        def __call__(self, state: str = "", meta: dict | None = None) -> None:
            self.updates.append({"state": state, **(meta or {})})

    tracker = FakeState()
    monkeypatch.setattr(process_batch, "update_state", tracker)
    process_batch.tracker = tracker  # type: ignore[attr-defined]
    return process_batch


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.mark.requires_models
@pytest.mark.usefixtures("require_real_models")
class TestBatchProcessing:
    """These run real inference end to end, so they need real artifacts.

    models/artifacts/ is gitignored deliberately - a 95 MB ONNX file does not
    belong in git - so on a fresh clone or in CI these skip rather than fail.
    Prepare them with: python scripts/prepare_models.py
    """

    def test_processes_every_item(self, worker_task, sample_image: bytes) -> None:
        items = [{"image_base64": b64(sample_image), "image_id": f"img-{i}"} for i in range(4)]
        summary = worker_task.run(
            job_id="job-1", task_type="classification", items=items, params={"top_k": 3}
        )

        assert summary["total_items"] == 4
        assert summary["completed_items"] == 4
        assert summary["failed_items"] == 0
        assert summary["status"] == "completed"
        assert len(summary["results"]) == 4

    def test_results_carry_index_and_image_id(self, worker_task, sample_image: bytes) -> None:
        items = [{"image_base64": b64(sample_image), "image_id": "photo-a"}]
        results = worker_task.run(job_id="j", task_type="classification", items=items, params={})[
            "results"
        ]
        assert results[0]["index"] == 0
        assert results[0]["image_id"] == "photo-a"
        assert results[0]["duration_ms"] >= 0

    def test_one_bad_image_does_not_fail_the_batch(self, worker_task, sample_image: bytes) -> None:
        """The headline guarantee: item 1 is junk, items 0 and 2 still succeed."""
        items = [
            {"image_base64": b64(sample_image), "image_id": "good-1"},
            {"image_base64": b64(b"this is not an image at all"), "image_id": "bad"},
            {"image_base64": b64(sample_image), "image_id": "good-2"},
        ]
        summary = worker_task.run(
            job_id="job-mixed", task_type="classification", items=items, params={}
        )

        assert summary["completed_items"] == 2
        assert summary["failed_items"] == 1
        assert summary["status"] == "completed"  # partial success is still success

        results = summary["results"]
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[1]["error"]["code"] == "INVALID_IMAGE"
        assert results[2]["success"] is True
        # The failing item must not have a result, and vice versa.
        assert results[1]["result"] is None
        assert results[0]["error"] is None

    def test_item_with_no_image_is_reported_not_raised(
        self, worker_task, sample_image: bytes
    ) -> None:
        items = [
            {"image_base64": b64(sample_image)},
            {"image_id": "no-image-here"},
        ]
        summary = worker_task.run(job_id="j", task_type="classification", items=items, params={})
        assert summary["completed_items"] == 1
        assert summary["failed_items"] == 1
        assert summary["results"][1]["error"]["code"] == "ITEM_PROCESSING_FAILED"

    def test_all_items_failing_reports_failed(self, worker_task) -> None:
        items = [{"image_base64": b64(b"garbage")} for _ in range(3)]
        summary = worker_task.run(job_id="j", task_type="classification", items=items, params={})
        assert summary["completed_items"] == 0
        assert summary["failed_items"] == 3
        assert summary["status"] == "failed"

    def test_detection_task(self, worker_task, sample_image: bytes) -> None:
        items = [{"image_base64": b64(sample_image)}]
        summary = worker_task.run(
            job_id="j",
            task_type="detection",
            items=items,
            params={"confidence_threshold": 0.5, "iou_threshold": 0.45},
        )
        assert summary["completed_items"] == 1
        assert "detections" in summary["results"][0]["result"]

    def test_similarity_task(self, worker_task, sample_image: bytes) -> None:
        items = [{"image_base64": b64(sample_image)}]
        summary = worker_task.run(job_id="j", task_type="similarity", items=items, params={})
        assert summary["results"][0]["result"]["dimension"] == 2048

    def test_respects_top_k_param(self, worker_task, sample_image: bytes) -> None:
        items = [{"image_base64": b64(sample_image)}]
        summary = worker_task.run(
            job_id="j", task_type="classification", items=items, params={"top_k": 7}
        )
        assert len(summary["results"][0]["result"]["predictions"]) == 7

    def test_reports_progress_after_each_item(self, worker_task, sample_image: bytes) -> None:
        """Progress must update per item, not jump from 0% to 100%."""
        items = [{"image_base64": b64(sample_image)} for _ in range(5)]
        worker_task.run(job_id="j", task_type="classification", items=items, params={})

        updates = worker_task.tracker.updates
        assert len(updates) == 5
        assert updates[0]["progress_percent"] == 20.0
        assert updates[-1]["progress_percent"] == 100.0

    def test_propagates_correlation_id(self, worker_task, sample_image: bytes) -> None:
        summary = worker_task.run(
            job_id="j",
            task_type="classification",
            items=[{"image_base64": b64(sample_image)}],
            params={},
            correlation_id="trace-abc-123",
        )
        assert summary["correlation_id"] == "trace-abc-123"

    def test_records_timing(self, worker_task, sample_image: bytes) -> None:
        summary = worker_task.run(
            job_id="j",
            task_type="classification",
            items=[{"image_base64": b64(sample_image)}],
            params={},
        )
        assert summary["duration_seconds"] >= 0
        assert summary["submitted_at"]
        assert summary["completed_at"]

    def test_empty_batch_does_not_divide_by_zero(self, worker_task) -> None:
        summary = worker_task.run(job_id="j", task_type="classification", items=[], params={})
        assert summary["total_items"] == 0
        assert summary["results"] == []


class TestHealthCheckTask:
    def test_returns_ok(self) -> None:
        from worker.tasks import health_check

        result = health_check.run()
        assert result["status"] == "ok"
        assert result["timestamp"]
