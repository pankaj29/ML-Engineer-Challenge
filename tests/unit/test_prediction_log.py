"""Recording predictions to the inference log.

This is the table drift detection, A/B testing and canary evaluation all read.
It existed, with an ORM model and a `log_inference` method, and no production
code path ever called it. Nothing failed: the table stayed empty, drift found
nothing to report, the retraining loop was told there was no drift, and every
layer above reported success.

So the first test here is the one that would have caught it: the routes call
this at all.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any

import pytest

from api.services.prediction_log import image_fingerprint, record_prediction

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Timing:
    preprocess_ms = 1.0
    inference_ms = 12.5
    postprocess_ms = 0.5
    total_ms = 14.0


class _Model:
    name = "resnet50"
    version = "1.0.0"
    runtime = "onnx"
    device = "cpu"


class _Prediction:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence


class _Response:
    correlation_id = "abc123"
    model = _Model()
    timing = _Timing()
    predictions = [_Prediction("cat", 0.9), _Prediction("dog", 0.05)]


@pytest.fixture
def captured(monkeypatch) -> list[dict[str, Any]]:
    """Capture what would be written, without a database."""
    rows: list[dict[str, Any]] = []

    class Recorder:
        available = True

        async def log_inference(self, record):
            rows.append(record)
            return 1

    monkeypatch.setattr("api.services.prediction_log.get_db_service", lambda: Recorder())
    return rows


class TestTheRoutesActuallyCallIt:
    """The regression that started all of this.

    Checked in the source rather than by running a request, because the write
    is fire-and-forget: a route could stop calling it and every response-level
    assertion would still pass.
    """

    @pytest.mark.parametrize("router", ["classification", "detection"])
    def test_every_prometheus_record_has_a_database_row_beside_it(self, router: str) -> None:
        source = (REPO_ROOT / "api" / "routers" / f"{router}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        metrics = calls.count("record_inference")
        rows = calls.count("record_prediction")

        assert metrics > 0, f"{router} records no metrics at all"
        assert rows == metrics, (
            f"{router} records {metrics} Prometheus metrics but writes {rows} "
            "database rows. The metric says how many and how fast; only the row "
            "says what the model predicted, which is what drift and A/B read."
        )


class TestTheRecord:
    async def test_it_captures_what_a_comparison_needs(self, captured) -> None:
        record_prediction(task="classification", response=_Response(), image_bytes=b"img")
        await asyncio.sleep(0)  # let the scheduled write run
        assert len(captured) == 1

        row = captured[0]
        # Version is the field that makes a canary evaluable at all.
        assert row["model_version"] == "1.0.0"
        assert row["model_name"] == "resnet50"
        assert row["task"] == "classification"
        assert row["top_label"] == "cat"
        assert row["top_confidence"] == pytest.approx(0.9)
        assert row["num_results"] == 2
        assert row["inference_ms"] == pytest.approx(12.5)

    async def test_the_image_is_hashed_not_stored(self, captured) -> None:
        """This table is queried for analytics and kept a long time. User
        images should not be in it."""
        image = b"\\xff\\xd8pretend jpeg bytes"
        record_prediction(task="classification", response=_Response(), image_bytes=image)
        await asyncio.sleep(0)

        row = captured[0]
        assert row["image_hash"] == image_fingerprint(image)
        assert len(row["image_hash"]) == 64
        assert image not in repr(row).encode()

    def test_the_same_image_hashes_the_same_way(self) -> None:
        assert image_fingerprint(b"x") == image_fingerprint(b"x")
        assert image_fingerprint(b"x") != image_fingerprint(b"y")

    async def test_an_empty_result_set_is_recorded_not_skipped(self, captured) -> None:
        """A detector finding nothing is a real observation, and a run of them
        is exactly the drift signal worth catching."""

        class Empty(_Response):
            predictions: list = []

        record_prediction(task="detection", response=Empty(), image_bytes=b"img")
        await asyncio.sleep(0)
        assert captured[0]["num_results"] == 0
        assert captured[0]["top_label"] is None


class TestItCannotBreakARequest:
    """The prediction is computed and the user is waiting. Nothing here is
    worth failing that for."""

    async def test_a_malformed_response_does_not_raise(self, captured) -> None:
        class Broken:
            correlation_id = "x"
            # no .model at all

        record_prediction(task="classification", response=Broken(), image_bytes=b"img")
        await asyncio.sleep(0)
        assert captured == []

    def test_an_unavailable_database_is_skipped_quietly(self, monkeypatch) -> None:
        class Down:
            available = False

            async def log_inference(self, record):  # pragma: no cover
                raise AssertionError("wrote to an unavailable database")

        monkeypatch.setattr("api.services.prediction_log.get_db_service", lambda: Down())
        record_prediction(task="classification", response=_Response(), image_bytes=b"img")

    def test_it_works_outside_an_event_loop(self, captured) -> None:
        """Called from a synchronous context there is no loop to schedule on.
        That is not an error, just nothing to do."""
        record_prediction(task="classification", response=_Response(), image_bytes=b"img")
