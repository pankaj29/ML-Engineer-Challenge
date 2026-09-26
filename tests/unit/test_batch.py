"""Unit tests for the batch endpoint and worker task logic.

Celery is replaced with a fake, so these run without a broker. What is tested
is our own logic: tier caps, rate-limit cost, queue-failure handling, status
mapping, and the rule that one bad image must not fail the whole batch.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest


class FakeAsyncResult:
    """Stands in for ``celery.result.AsyncResult``."""

    def __init__(self, task_id: str, state: str = "PENDING", result: Any = None, info: Any = None):
        self.id = task_id
        self.state = state
        self.result = result
        self.info = info if info is not None else result
        self.revoked = False
        self.backend = self

    def store_result(self, *args: Any, **kwargs: Any) -> None:
        pass

    def revoke(self, **kwargs: Any) -> None:
        self.revoked = True


@pytest.fixture
def mock_celery(monkeypatch):
    """Replace Celery's enqueue and result lookup with in-memory fakes.

    The router enqueues with ``celery_app.send_task`` (by task name) rather
    than importing the task, so that is what is patched here. Patching the
    task object itself would test an implementation the router no longer uses.
    """
    state: dict[str, Any] = {"submitted": [], "results": {}, "fail_enqueue": False}

    from worker.celery_app import celery_app

    def fake_send_task(name: str, **kwargs: Any) -> FakeAsyncResult:
        if state["fail_enqueue"]:
            raise ConnectionError("broker unreachable")
        state["submitted"].append({"name": name, **kwargs})
        return FakeAsyncResult(kwargs.get("task_id", "fake-job-id"))

    monkeypatch.setattr(celery_app, "send_task", fake_send_task)

    def fake_async_result(job_id: str, app: Any = None) -> FakeAsyncResult:
        return state["results"].get(job_id, FakeAsyncResult(job_id))

    monkeypatch.setattr("celery.result.AsyncResult", fake_async_result)
    return state


@pytest.fixture
def batch_payload(sample_image_b64: str):
    def _build(count: int = 3, task: str = "classification") -> dict:
        return {
            "task": task,
            "items": [
                {"image_base64": sample_image_b64, "image_id": f"img-{i}"} for i in range(count)
            ],
        }

    return _build


class TestBatchRecordOrdering:
    """The row must exist before the worker can see the job."""

    @pytest.fixture
    def db_calls(self, monkeypatch, mock_celery):
        from worker.celery_app import celery_app

        calls: list[tuple] = []

        class RecordingDb:
            async def create_batch_job(self, record: dict) -> bool:
                calls.append(("create", record["status"]))
                return True

            async def update_batch_job(self, job_id: str, **fields: Any) -> bool:
                calls.append(("update", fields.get("status")))
                return True

        monkeypatch.setattr("api.services.db_service.get_db_service", lambda: RecordingDb())
        original = celery_app.send_task

        def send_task(name: str, **kwargs: Any):
            calls.append(("enqueue", None))
            return original(name, **kwargs)

        monkeypatch.setattr(celery_app, "send_task", send_task)
        return calls

    def test_row_is_written_before_the_job_is_enqueued(
        self, api_client, auth_headers, batch_payload, db_calls
    ) -> None:
        response = api_client.post("/api/v1/batch", json=batch_payload(2), headers=auth_headers)
        assert response.status_code == 202
        assert db_calls == [("create", "pending"), ("enqueue", None)]

    def test_failed_enqueue_marks_the_row_failed(
        self, api_client, auth_headers, batch_payload, mock_celery, db_calls
    ) -> None:
        mock_celery["fail_enqueue"] = True
        response = api_client.post("/api/v1/batch", json=batch_payload(2), headers=auth_headers)
        assert response.status_code == 503
        assert db_calls == [("create", "pending"), ("enqueue", None), ("update", "failed")]


class TestBatchSubmission:
    def test_returns_202_with_job_id(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        response = api_client.post("/api/v1/batch", json=batch_payload(3), headers=auth_headers)
        assert response.status_code == 202
        body = response.json()
        # The router generates the id (a UUID) so the same value reaches the
        # database, the queue and the caller.
        assert body["job_id"]
        assert body["status"] == "pending"
        assert body["total_items"] == 3
        assert body["status_url"].endswith(body["job_id"])

    def test_estimates_completion_time(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        body = api_client.post("/api/v1/batch", json=batch_payload(10), headers=auth_headers).json()
        assert body["estimated_seconds"] > 0

    def test_detection_is_estimated_slower_than_classification(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        fast = api_client.post(
            "/api/v1/batch", json=batch_payload(5, "classification"), headers=auth_headers
        ).json()
        slow = api_client.post(
            "/api/v1/batch", json=batch_payload(5, "detection"), headers=auth_headers
        ).json()
        assert slow["estimated_seconds"] > fast["estimated_seconds"]

    def test_passes_items_to_the_worker(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        api_client.post("/api/v1/batch", json=batch_payload(4), headers=auth_headers)
        sent = mock_celery["submitted"][0]
        assert sent["name"] == "worker.tasks.process_batch"
        kwargs = sent["kwargs"]
        assert len(kwargs["items"]) == 4
        assert kwargs["task_type"] == "classification"
        # The correlation id must travel to the worker for end-to-end tracing.
        assert kwargs["correlation_id"]
        # The job id sent to the worker must match the one handed to the caller.
        assert kwargs["job_id"] == sent["task_id"]

    def test_requires_authentication(self, api_client, mock_celery, batch_payload) -> None:
        assert api_client.post("/api/v1/batch", json=batch_payload(2)).status_code == 401

    def test_free_tier_batch_cap(
        self, api_client, free_tier_headers, mock_celery, batch_payload
    ) -> None:
        assert (
            api_client.post(
                "/api/v1/batch", json=batch_payload(5), headers=free_tier_headers
            ).status_code
            == 202
        )
        response = api_client.post(
            "/api/v1/batch", json=batch_payload(6), headers=free_tier_headers
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "BATCH_TOO_LARGE"
        assert response.json()["error"]["details"]["limit"] == 5

    def test_pro_tier_allows_more(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        assert (
            api_client.post(
                "/api/v1/batch", json=batch_payload(20), headers=auth_headers
            ).status_code
            == 202
        )

    def test_rejects_empty_batch(self, api_client, auth_headers, mock_celery) -> None:
        response = api_client.post(
            "/api/v1/batch", json={"task": "classification", "items": []}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_rejects_absurd_batch(
        self, api_client, auth_headers, mock_celery, sample_image_b64
    ) -> None:
        response = api_client.post(
            "/api/v1/batch",
            json={
                "task": "classification",
                "items": [{"image_base64": sample_image_b64}] * 500,
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_rejects_invalid_task(
        self, api_client, auth_headers, mock_celery, sample_image_b64
    ) -> None:
        response = api_client.post(
            "/api/v1/batch",
            json={"task": "teleportation", "items": [{"image_base64": sample_image_b64}]},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_rejects_non_https_callback(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        payload = batch_payload(2)
        payload["callback_url"] = "http://insecure.example.com/hook"
        assert (
            api_client.post("/api/v1/batch", json=payload, headers=auth_headers).status_code == 422
        )

    def test_accepts_https_callback(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        payload = batch_payload(2)
        payload["callback_url"] = "https://secure.example.com/hook"
        assert (
            api_client.post("/api/v1/batch", json=payload, headers=auth_headers).status_code == 202
        )

    def test_broker_down_returns_503_not_500(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        """A dead queue is a service problem with a clear message, not a crash."""
        mock_celery["fail_enqueue"] = True
        response = api_client.post("/api/v1/batch", json=batch_payload(2), headers=auth_headers)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"

    def test_unknown_model_rejected_before_queueing(
        self, api_client, auth_headers, mock_celery, batch_payload
    ) -> None:
        """Fail fast rather than queue work that is certain to fail."""
        payload = batch_payload(2)
        payload["model_name"] = "no-such-model"
        assert (
            api_client.post("/api/v1/batch", json=payload, headers=auth_headers).status_code == 404
        )
        assert mock_celery["submitted"] == []


class TestBatchStatus:
    def test_unknown_job_returns_404(self, api_client, auth_headers, mock_celery) -> None:
        response = api_client.get("/api/v1/batch/never-existed", headers=auth_headers)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "JOB_NOT_FOUND"

    def test_running_job_reports_progress(self, api_client, auth_headers, mock_celery) -> None:
        mock_celery["results"]["running-job"] = FakeAsyncResult(
            "running-job",
            state="PROGRESS",
            info={"completed": 7, "failed": 1, "total": 20, "progress_percent": 40.0},
        )
        body = api_client.get("/api/v1/batch/running-job", headers=auth_headers).json()
        assert body["status"] == "running"
        assert body["completed_items"] == 7
        assert body["failed_items"] == 1
        assert body["progress_percent"] == 40.0

    def test_completed_job_returns_results(self, api_client, auth_headers, mock_celery) -> None:
        mock_celery["results"]["done-job"] = FakeAsyncResult(
            "done-job",
            state="SUCCESS",
            result={
                "completed_items": 2,
                "failed_items": 1,
                "total_items": 3,
                "duration_seconds": 1.5,
                "completed_at": datetime.now(UTC).isoformat(),
                "results": [
                    {
                        "index": 0,
                        "image_id": "a",
                        "success": True,
                        "result": {"x": 1},
                        "duration_ms": 10.0,
                    },
                    {
                        "index": 1,
                        "image_id": "b",
                        "success": True,
                        "result": {"x": 2},
                        "duration_ms": 12.0,
                    },
                    {
                        "index": 2,
                        "image_id": "c",
                        "success": False,
                        "error": {"code": "INVALID_IMAGE", "message": "bad"},
                        "duration_ms": 1.0,
                    },
                ],
            },
        )
        body = api_client.get("/api/v1/batch/done-job", headers=auth_headers).json()

        assert body["status"] == "completed"
        assert body["progress_percent"] == 100.0
        assert len(body["results"]) == 3
        # One failed item must not hide the two that succeeded.
        assert body["results"][2]["success"] is False
        assert body["results"][2]["error"]["code"] == "INVALID_IMAGE"

    def test_failed_job_hides_internal_detail(self, api_client, auth_headers, mock_celery) -> None:
        mock_celery["results"]["bad-job"] = FakeAsyncResult(
            "bad-job", state="FAILURE", info=RuntimeError("/srv/app/worker.py line 42 exploded")
        )
        body = api_client.get("/api/v1/batch/bad-job", headers=auth_headers).json()
        assert body["status"] == "failed"
        assert "/srv/app" not in body["error"]

    def test_results_can_be_omitted(self, api_client, auth_headers, mock_celery) -> None:
        mock_celery["results"]["done2"] = FakeAsyncResult(
            "done2",
            state="SUCCESS",
            result={
                "completed_items": 1,
                "failed_items": 0,
                "total_items": 1,
                "results": [{"index": 0, "success": True, "result": {}, "duration_ms": 1.0}],
            },
        )
        body = api_client.get(
            "/api/v1/batch/done2?include_results=false", headers=auth_headers
        ).json()
        assert body["results"] is None


class TestBatchCancellation:
    def test_cancels_running_job(self, api_client, auth_headers, mock_celery) -> None:
        job = FakeAsyncResult("cancel-me", state="PROGRESS")
        mock_celery["results"]["cancel-me"] = job

        body = api_client.delete("/api/v1/batch/cancel-me", headers=auth_headers).json()
        assert body["cancelled"] is True
        assert job.revoked is True

    def test_cannot_cancel_finished_job(self, api_client, auth_headers, mock_celery) -> None:
        mock_celery["results"]["finished"] = FakeAsyncResult("finished", state="SUCCESS", result={})
        body = api_client.delete("/api/v1/batch/finished", headers=auth_headers).json()
        assert body["cancelled"] is False
        assert "already finished" in body["reason"]


class TestBatchOwnership:
    """A job id alone must not reach another user's results or cancel their job."""

    @pytest.fixture
    def owned_by(self, monkeypatch):
        from types import SimpleNamespace

        from api.middleware.auth import _key_fingerprint

        class FakeDb:
            available = True

            def __init__(self) -> None:
                self.records: dict[str, Any] = {}
                self.updated: list[str] = []

            async def get_batch_job(self, job_id: str) -> Any:
                return self.records.get(job_id)

            async def update_batch_job(self, job_id: str, **fields: Any) -> bool:
                self.updated.append(job_id)
                return True

        db = FakeDb()
        monkeypatch.setattr("api.services.db_service.get_db_service", lambda: db)

        def _record(job_id: str, api_key: str) -> FakeDb:
            db.records[job_id] = SimpleNamespace(
                user_id=f"key_{_key_fingerprint(api_key)}",
                total_items=1,
                task="classification",
                submitted_at=datetime.now(UTC),
                started_at=None,
                completed_at=None,
                status="pending",
                completed_items=0,
                failed_items=0,
            )
            return db

        return _record

    def test_owner_can_read_their_job(
        self, api_client, auth_headers, mock_celery, owned_by
    ) -> None:
        owned_by("mine", "test-pro-key")
        mock_celery["results"]["mine"] = FakeAsyncResult("mine", state="SUCCESS", result={})
        assert api_client.get("/api/v1/batch/mine", headers=auth_headers).status_code == 200

    def test_other_users_job_reads_as_not_found(
        self, api_client, auth_headers, mock_celery, owned_by
    ) -> None:
        owned_by("theirs", "someone-elses-key")
        mock_celery["results"]["theirs"] = FakeAsyncResult("theirs", state="SUCCESS", result={})
        response = api_client.get("/api/v1/batch/theirs", headers=auth_headers)
        assert response.status_code == 404

    def test_other_users_job_cannot_be_cancelled(
        self, api_client, auth_headers, mock_celery, owned_by
    ) -> None:
        db = owned_by("theirs", "someone-elses-key")
        job = FakeAsyncResult("theirs", state="PROGRESS")
        mock_celery["results"]["theirs"] = job
        response = api_client.delete("/api/v1/batch/theirs", headers=auth_headers)
        assert response.status_code == 404
        assert job.revoked is False
        assert db.updated == []

    def test_expired_result_falls_back_to_the_recorded_outcome(
        self, api_client, auth_headers, mock_celery, owned_by
    ) -> None:
        """After Celery's 24 h expiry it reports PENDING; the row says otherwise."""
        db = owned_by("old", "test-pro-key")
        db.records["old"].status = "completed"
        db.records["old"].completed_items = 2
        db.records["old"].failed_items = 1
        body = api_client.get("/api/v1/batch/old", headers=auth_headers).json()
        assert body["status"] == "completed"
        assert (body["completed_items"], body["failed_items"]) == (2, 1)

    def test_unknown_id_is_not_revoked(
        self, api_client, auth_headers, mock_celery, owned_by
    ) -> None:
        owned_by("unrelated", "test-pro-key")
        job = FakeAsyncResult("never-submitted", state="PENDING")
        mock_celery["results"]["never-submitted"] = job
        response = api_client.delete("/api/v1/batch/never-submitted", headers=auth_headers)
        assert response.status_code == 404
        assert job.revoked is False


class TestWorkerItemDecoding:
    """The worker's per-item decoding, tested without Celery."""

    def test_decodes_base64(self, sample_image_b64: str) -> None:
        from worker.tasks import _decode_item

        assert _decode_item({"image_base64": sample_image_b64})[:4] == b"\x89PNG"

    def test_rejects_item_with_no_image(self) -> None:
        from worker.tasks import _decode_item

        with pytest.raises(ValueError, match="neither"):
            _decode_item({"image_id": "orphan"})

    def test_url_item_is_ssrf_checked(self) -> None:
        """The worker runs outside the API's request path and must re-check."""
        from api.exceptions import ValidationError
        from worker.tasks import _decode_item

        with pytest.raises(ValidationError):
            _decode_item({"image_url": "http://169.254.169.254/latest/meta-data/"})
