"""The retraining gates, running the real validators.

The unit tests mock `_run_validation` and `_run_regression`, so they prove the
policy reacts correctly to a pass or a fail but say nothing about whether the
pipeline can call the real thing. A signature change in `validate_model` would
leave every one of them green and the pipeline broken.

These call the real functions against the real registered model. Training is
stubbed out, because an epoch on 100k images is not a unit of work a test
suite should contain, and because training is covered on its own. Everything
after it is genuine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.pipeline.retraining import Stage, run_pipeline

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.requires_models]

MODEL = "resnet50-tiny-imagenet"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _artifact_present() -> bool:
    path = REPO_ROOT / "models" / "artifacts" / "resnet50-tiny-imagenet.onnx"
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return not handle.read(42).startswith(b"version https://git-lfs")


pytestmark.append(
    pytest.mark.skipif(not _artifact_present(), reason="the ONNX artifact is not checked out")
)


def drift_report() -> dict:
    """High-severity drift, so the policy says retrain and the gates run."""
    return {
        "model": MODEL,
        "overall_severity": "high",
        "drifted": True,
        "results": [{"test": "ks", "feature": "confidence", "drifted": True, "effect_size": 0.45}],
    }


@pytest.fixture
def no_training(monkeypatch):
    """Skip training, keep every gate after it real.

    The artifacts on disk are the ones the gates will judge, which is exactly
    the case that matters: a retrain that produced this model would have to
    pass on its own merits.
    """
    monkeypatch.setattr("models.pipeline.retraining._run_training", lambda **kwargs: True)


class TestTheRealGates:
    def test_the_pipeline_can_call_the_real_validator(self, no_training, tmp_path) -> None:
        """Not that it passes, that the call works.

        A mocked gate would keep passing after `validate_model` changed shape.
        This fails loudly if the pipeline can no longer reach it.
        """
        result = run_pipeline(
            model=MODEL,
            drift_report=drift_report(),
            data_dir=REPO_ROOT / "data",
            output_dir=tmp_path,
            history_path=tmp_path / "history.json",
            dry_run=False,
            epochs=1,
            track="none",
        )

        assert result.stage is not Stage.ASSESSED, "the pipeline stopped before training"
        assert (
            result.validation is not None
        ), "validation returned nothing, so the pipeline could not reach the real validator"
        assert "passed" in result.validation
        assert isinstance(result.validation.get("checks"), list)

    def test_the_outcome_is_consistent_with_the_gates(self, no_training, tmp_path) -> None:
        """Whatever the real validators say, the pipeline must agree with it.

        Promoting a model whose validation failed is the bug this whole
        section exists to prevent, and it is the one a mock cannot catch.
        """
        result = run_pipeline(
            model=MODEL,
            drift_report=drift_report(),
            data_dir=REPO_ROOT / "data",
            output_dir=tmp_path,
            history_path=tmp_path / "history.json",
            dry_run=False,
            epochs=1,
            track="none",
        )

        validation_passed = bool((result.validation or {}).get("passed"))
        regression_passed = result.regression is None or bool(result.regression.get("passed"))

        if validation_passed and regression_passed:
            assert result.promoted is True
            assert result.stage is Stage.PROMOTED
        else:
            assert result.promoted is False, (
                f"promoted despite validation={validation_passed} "
                f"regression={regression_passed}"
            )
            assert result.stage is Stage.FAILED

    def test_the_run_is_recorded_whatever_happened(self, no_training, tmp_path) -> None:
        history = tmp_path / "history.json"
        run_pipeline(
            model=MODEL,
            drift_report=drift_report(),
            data_dir=REPO_ROOT / "data",
            output_dir=tmp_path,
            history_path=history,
            dry_run=False,
            epochs=1,
            track="none",
        )

        entries = json.loads(history.read_text(encoding="utf-8"))
        assert len(entries) == 1
        entry = entries[0]
        assert entry["stage"] in {s.value for s in Stage}
        assert entry["decision"]["should_retrain"] is True
        assert entry["dry_run"] is False

    def test_a_real_run_starts_the_cooldown(self, no_training, tmp_path) -> None:
        """The second call must be refused, or a persistent drift signal would
        retrain on every scheduled run."""
        history = tmp_path / "history.json"
        first = run_pipeline(
            model=MODEL,
            drift_report=drift_report(),
            data_dir=REPO_ROOT / "data",
            output_dir=tmp_path,
            history_path=history,
            dry_run=False,
            epochs=1,
            track="none",
        )
        assert first.stage is not Stage.SKIPPED

        second = run_pipeline(
            model=MODEL,
            drift_report=drift_report(),
            data_dir=REPO_ROOT / "data",
            output_dir=tmp_path,
            history_path=history,
            dry_run=False,
            epochs=1,
            track="none",
        )
        assert second.stage is Stage.SKIPPED
        assert second.decision["blocked_by"] == "cooldown"
