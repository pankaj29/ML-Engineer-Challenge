"""Every INT8 artifact must still do its job, checked against its fp32 twin.

The INT8 YOLO model once shipped emitting a class score of 0.0 for every
anchor: its head concatenates box coordinates (0-640) and class scores (0-1),
and quantizing that Concat gave both one int8 scale. It detected nothing, and
the quantization report said "100% agreement" because agreement was only
computed for classifier-shaped outputs. These tests run each registered INT8
model on the committed sample images, through the serving path, and compare
it with fp32.

CI runs this file with REQUIRE_MODELS=1, so missing artifacts fail rather than
skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from api.models.schemas import RuntimeFormat, TaskType

pytestmark = [pytest.mark.integration, pytest.mark.requires_models, pytest.mark.slow]

SAMPLES = sorted(Path("samples").glob("*.jpg"))


@pytest.fixture
def inference(require_real_models: None):
    from api.config import Settings
    from api.services.cache_service import CacheService
    from api.services.inference_service import InferenceService
    from api.services.model_service import ModelService

    cfg = Settings(environment="test", cache_enabled=False)
    models = ModelService(cfg)
    yield InferenceService(models, CacheService(cfg), cfg)
    models.unload_all()


def _int8_entries(inference) -> list:
    entries = [
        e
        for e in inference.models.list_entries()
        if RuntimeFormat.ONNX_INT8.value in e.artifacts and RuntimeFormat.ONNX.value in e.artifacts
    ]
    assert entries, "no model registers both an fp32 and an INT8 ONNX artifact"
    return entries


async def _both(inference, entry, image: bytes):
    call = {
        TaskType.CLASSIFICATION: lambda rt: inference.classify(
            image, top_k=5, model_name=entry.name, model_version=entry.version, runtime=rt
        ),
        TaskType.DETECTION: lambda rt: inference.detect(
            image, model_name=entry.name, model_version=entry.version, runtime=rt
        ),
        TaskType.SIMILARITY: lambda rt: inference.embed(
            image, model_name=entry.name, model_version=entry.version, runtime=rt
        ),
    }[entry.task]
    fp32 = await call(RuntimeFormat.ONNX)
    int8 = await call(RuntimeFormat.ONNX_INT8)
    # A silent fallback to fp32 would make every comparison below trivially pass.
    assert int8.model.runtime == RuntimeFormat.ONNX_INT8, f"{entry.key} INT8 fell back"
    return fp32, int8


def test_samples_are_present() -> None:
    assert len(SAMPLES) >= 3


async def test_int8_agrees_with_fp32_on_every_sample(inference) -> None:
    failures: list[str] = []
    for entry in _int8_entries(inference):
        for path in SAMPLES:
            fp32, int8 = await _both(inference, entry, path.read_bytes())
            where = f"{entry.key} on {path.name}"

            if entry.task == TaskType.DETECTION:
                fp32_labels = {d.label for d in fp32.detections}
                int8_labels = {d.label for d in int8.detections}
                if fp32_labels and not int8_labels:
                    failures.append(
                        f"{where}: fp32 found {sorted(fp32_labels)}, INT8 found nothing"
                    )
                elif fp32.detections:
                    top = max(fp32.detections, key=lambda d: d.confidence).label
                    if top not in int8_labels:
                        failures.append(f"{where}: INT8 missed fp32's top label {top!r}")

            elif entry.task == TaskType.CLASSIFICATION:
                # Loose on purpose: INT8 top-1 agreement on the fine-tuned
                # model is well below 100%, but fp32's answer leaving the
                # top 5 means the model is broken, not just less precise.
                top1 = fp32.predictions[0].label
                if top1 not in {p.label for p in int8.predictions}:
                    failures.append(f"{where}: fp32 top-1 {top1!r} not in INT8 top-5")

            else:
                a = np.asarray(fp32.embedding, dtype=np.float64)
                b = np.asarray(int8.embedding, dtype=np.float64)
                cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                if cosine < 0.9:
                    failures.append(f"{where}: embedding cosine to fp32 is {cosine:.3f}")

    assert not failures, "INT8 models diverge from fp32:\n" + "\n".join(failures)
