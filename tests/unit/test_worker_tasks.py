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


class TestFetchingAnImageByUrl:
    """`image_url` items go through the same SSRF checks as the API.

    A worker that fetched arbitrary URLs would be a way around every
    protection on the synchronous endpoint, since it runs inside the network
    and its input arrives from a queue rather than a request.
    """

    def test_a_url_item_is_downloaded(self, monkeypatch) -> None:
        from worker import tasks

        class FakeResponse:
            content = b"\xff\xd8image bytes"

            def raise_for_status(self) -> None:
                return None

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                captured["client"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url):
                captured["url"] = url
                return FakeResponse()

        captured: dict = {}
        monkeypatch.setitem(
            __import__("sys").modules, "httpx", type("M", (), {"Client": FakeClient})
        )

        assert tasks._decode_item({"image_url": "https://example.com/a.jpg"}) == (
            b"\xff\xd8image bytes"
        )
        assert captured["url"] == "https://example.com/a.jpg"
        # Redirects are the standard way round an SSRF allowlist: the first
        # hop passes validation and the second goes wherever it likes.
        assert captured["client"]["follow_redirects"] is False
        assert captured["client"]["timeout"] == 10.0

    def test_a_url_the_validator_rejects_never_gets_fetched(self, monkeypatch) -> None:
        from worker import tasks

        def explode(**kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("a rejected URL was fetched anyway")

        monkeypatch.setitem(__import__("sys").modules, "httpx", type("M", (), {"Client": explode}))

        with pytest.raises(Exception):  # noqa: B017 - the validator's own type
            tasks._decode_item({"image_url": "http://169.254.169.254/latest/meta-data/"})

    def test_an_item_with_neither_field_is_refused(self) -> None:
        from worker import tasks

        with pytest.raises(ValueError, match="neither image_base64 nor image_url"):
            tasks._decode_item({"image_id": "nothing-here"})


class TestTheCompletionCallback:
    """Best-effort by design: the results are already stored and pollable, so
    a callback failure must never fail the job."""

    def test_the_callback_is_posted_with_the_summary(self, monkeypatch) -> None:
        from worker import tasks

        posted: dict = {}

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                posted["client"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, json):
                posted["url"] = url
                posted["payload"] = json
                return type("R", (), {"status_code": 200})()

        monkeypatch.setitem(
            __import__("sys").modules, "httpx", type("M", (), {"Client": FakeClient})
        )

        tasks._post_callback("https://example.com/done", {"job_id": "j1", "total": 2})
        assert posted["url"] == "https://example.com/done"
        assert posted["payload"]["job_id"] == "j1"
        assert posted["client"]["follow_redirects"] is False

    def test_a_failing_callback_does_not_raise(self, monkeypatch) -> None:
        from worker import tasks

        class Exploding:
            def __init__(self, **kwargs) -> None:
                raise ConnectionError("the caller's endpoint is down")

        monkeypatch.setitem(
            __import__("sys").modules, "httpx", type("M", (), {"Client": Exploding})
        )

        # No exception: the job succeeded and its results are retrievable.
        tasks._post_callback("https://example.com/done", {"job_id": "j1"})

    def test_a_finished_job_notifies_the_caller(
        self, worker_task, sample_image: bytes, monkeypatch
    ) -> None:
        """The call site, not just the helper. A callback_url that is accepted
        and then never used is the kind of gap nobody notices until someone
        waits for a notification that is not coming."""
        from worker import tasks

        sent: list[dict] = []
        monkeypatch.setattr(
            tasks,
            "_post_callback",
            lambda url, payload: sent.append({"url": url, "payload": payload}),
        )

        items = [{"image_base64": b64(sample_image), "image_id": "a"}]
        summary = worker_task.run(
            job_id="cb-job",
            task_type="classification",
            items=items,
            params={"callback_url": "https://example.com/done"},
        )

        assert len(sent) == 1, "the job finished without calling back"
        assert sent[0]["url"] == "https://example.com/done"
        assert sent[0]["payload"]["job_id"] == "cb-job"
        assert sent[0]["payload"] is summary, "the callback got something other than the summary"

    def test_no_callback_url_means_no_call(
        self, worker_task, sample_image: bytes, monkeypatch
    ) -> None:
        from worker import tasks

        def explode(url, payload):  # pragma: no cover - must not be reached
            raise AssertionError("called back without a callback_url")

        monkeypatch.setattr(tasks, "_post_callback", explode)
        worker_task.run(
            job_id="no-cb",
            task_type="classification",
            items=[{"image_base64": b64(sample_image), "image_id": "a"}],
            params={},
        )

    def test_a_callback_url_that_fails_validation_is_swallowed(self, monkeypatch) -> None:
        """Same SSRF rules as image fetching, and still not fatal."""
        from worker import tasks

        tasks._post_callback("http://169.254.169.254/", {"job_id": "j1"})


class TestSoftTimeout:
    """When the job runs out of time, keep what was finished.

    Partial results are far more useful than none, and the caller needs to be
    able to tell which items were skipped rather than guessing from a short
    list.
    """

    def test_a_timeout_keeps_completed_work_and_marks_the_rest(
        self, worker_task, sample_image: bytes, monkeypatch
    ) -> None:
        from celery.exceptions import SoftTimeLimitExceeded

        from worker import tasks

        calls = {"n": 0}
        real_run_one = tasks._run_one

        async def fail_on_the_third(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise SoftTimeLimitExceeded
            return await real_run_one(*args, **kwargs)

        monkeypatch.setattr(tasks, "_run_one", fail_on_the_third)

        items = [{"image_base64": b64(sample_image), "image_id": f"i{i}"} for i in range(5)]
        summary = worker_task.run(
            job_id="timeout-job", task_type="classification", items=items, params={}
        )

        assert summary["completed_items"] == 2, "finished work was discarded"
        assert summary["failed_items"] >= 1

        timed_out = [r for r in summary["results"] if not r["success"]]
        assert timed_out, "nothing recorded the timeout"
        assert timed_out[0]["error"]["code"] == "JOB_TIMEOUT"

        # It stops rather than grinding through the remainder.
        assert len(summary["results"]) < len(items)


class TestUnsupportedTask:
    """An unrecognised task must stop the job, not guess at one.

    Silently classifying a detection request would return confident nonsense
    and the caller would have no way to tell.
    """

    def test_an_unknown_task_type_is_refused(self, worker_task, sample_image: bytes) -> None:
        items = [{"image_base64": b64(sample_image), "image_id": "x"}]
        with pytest.raises(ValueError, match="segmentation"):
            worker_task.run(job_id="bad-task", task_type="segmentation", items=items, params={})

    async def test_a_task_type_the_dispatcher_forgot_is_refused(self) -> None:
        """The guard inside `_run_one`, for an enum member added without a
        matching branch. Unreachable from the task entry point because
        TaskType() rejects the string first, so it is called directly."""
        from worker import tasks

        class FutureTask:
            value = "segmentation"

            def __str__(self) -> str:
                return "TaskType.SEGMENTATION"

        with pytest.raises(ValueError, match="unsupported task type"):
            await tasks._run_one(None, FutureTask(), b"bytes", {})
