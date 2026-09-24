"""Unit tests for exponential moving averaging of the weights.

EMA is the kind of feature that fails silently. If the decay warmup is wrong
the average stays pinned near the initial weights and the reported accuracy is
noise; if buffers are averaged instead of copied the BatchNorm statistics lag
the real distribution; if the average is not checkpointed a resumed run throws
away every step of smoothing without saying so. None of those raise. They just
produce a slightly worse model and a number nobody can explain.

So these tests pin the arithmetic, not just that the code runs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.training.train_classifier import ModelEma


def _model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Linear(4, 2),
    )


def _step(model: torch.nn.Module, scale: float = 0.1) -> None:
    """Move every weight by a fixed amount, standing in for an optimiser step."""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(scale)


class TestTheAverageTracksTheWeights:
    def test_starts_as_a_copy(self) -> None:
        model = _model()
        ema = ModelEma(model)
        for a, b in zip(model.parameters(), ema.module.parameters(), strict=True):
            assert torch.allclose(a, b)

    def test_it_is_a_copy_and_not_a_reference(self) -> None:
        """Sharing storage would make the average identical to the weights."""
        model = _model()
        ema = ModelEma(model)
        before = ema.module[0].weight.clone()
        _step(model, 1.0)
        assert torch.allclose(ema.module[0].weight, before)

    def test_one_update_moves_the_average_toward_the_weights(self) -> None:
        model = _model()
        ema = ModelEma(model, decay=0.9)
        start = ema.module[0].weight.clone()
        _step(model, 1.0)
        ema.update(model)
        moved = ema.module[0].weight
        assert not torch.allclose(moved, start)
        assert torch.all(moved < model[0].weight), "the average overshot the live weights"

    def test_the_arithmetic_is_exactly_the_ema_formula(self) -> None:
        model = _model()
        ema = ModelEma(model, decay=0.5)
        start = ema.module[0].weight.clone()
        _step(model, 1.0)
        live = model[0].weight.clone()

        ema.update(model)

        decay = ema._current_decay()
        expected = start * decay + live * (1.0 - decay)
        assert torch.allclose(ema.module[0].weight, expected, atol=1e-6)

    def test_it_converges_on_stationary_weights(self) -> None:
        """Held still, the average must arrive at the weights, not drift."""
        model = _model()
        ema = ModelEma(model, decay=0.9)
        _step(model, 1.0)
        for _ in range(500):
            ema.update(model)
        assert torch.allclose(ema.module[0].weight, model[0].weight, atol=1e-3)


class TestDecayWarmup:
    def test_the_first_steps_use_a_low_decay(self) -> None:
        """A fixed 0.9998 from step 1 leaves the average stuck at init."""
        ema = ModelEma(_model(), decay=0.9998)
        ema.steps = 1
        assert ema._current_decay() < 0.3

    def test_the_decay_rises_toward_the_target(self) -> None:
        ema = ModelEma(_model(), decay=0.9998)
        ema.steps = 100
        early = ema._current_decay()
        ema.steps = 100_000
        assert early < ema._current_decay() <= 0.9998

    def test_it_never_exceeds_the_configured_decay(self) -> None:
        ema = ModelEma(_model(), decay=0.99)
        ema.steps = 10_000_000
        assert ema._current_decay() == pytest.approx(0.99)

    def test_the_average_is_useful_within_one_epoch(self) -> None:
        """The property the warmup exists for.

        Without it, ~500 steps of a 0.9998 decay leaves the average about 90%
        initial weights, so an early validation score means nothing.
        """
        model = _model()
        ema = ModelEma(model, decay=0.9998)
        _step(model, 1.0)
        for _ in range(500):
            ema.update(model)
        gap = (ema.module[0].weight - model[0].weight).abs().max()
        assert gap < 0.1, f"average still {gap:.3f} away from the weights after 500 steps"


class TestBuffersAreCopiedNotAveraged:
    def test_batchnorm_statistics_track_the_live_model_exactly(self) -> None:
        """Running stats are already averages; averaging them again lags."""
        model = _model()
        ema = ModelEma(model, decay=0.9)
        with torch.no_grad():
            model[1].running_mean.fill_(5.0)
        ema.update(model)
        assert torch.allclose(ema.module[1].running_mean, torch.full((4,), 5.0))

    def test_integer_buffers_survive_intact(self) -> None:
        """`num_batches_tracked` is an int64; averaging truncates it."""
        model = _model()
        ema = ModelEma(model, decay=0.9)
        with torch.no_grad():
            model[1].num_batches_tracked.fill_(17)
        ema.update(model)
        assert ema.module[1].num_batches_tracked.item() == 17
        assert not ema.module[1].num_batches_tracked.dtype.is_floating_point


class TestTheAverageDoesNotTrain:
    def test_its_parameters_require_no_gradient(self) -> None:
        ema = ModelEma(_model())
        assert not any(p.requires_grad for p in ema.module.parameters())

    def test_it_is_held_in_eval_mode(self) -> None:
        """In train mode a forward pass would mutate its BatchNorm stats."""
        model = _model()
        model.train()
        assert ModelEma(model).module.training is False


class TestRoundTrip:
    def test_state_survives_a_save_and_restore(self, tmp_path) -> None:
        """Resume must not silently discard the smoothing done so far."""
        model = _model()
        ema = ModelEma(model, decay=0.9)
        for _ in range(20):
            _step(model, 0.05)
            ema.update(model)

        path = tmp_path / "ema.pt"
        torch.save(ema.state_dict(), path)

        restored = ModelEma(_model(seed=1), decay=0.5)
        restored.load_state_dict(torch.load(path, weights_only=False))

        assert restored.steps == 20
        assert restored.decay == pytest.approx(0.9)
        assert torch.allclose(restored.module[0].weight, ema.module[0].weight)

    def test_the_step_count_carries_so_the_warmup_is_not_restarted(self, tmp_path) -> None:
        ema = ModelEma(_model(), decay=0.9998)
        ema.steps = 5_000
        restored = ModelEma(_model(), decay=0.9998)
        restored.load_state_dict(ema.state_dict())
        assert restored._current_decay() == pytest.approx(ema._current_decay())


# ---------------------------------------------------------------------------
# Integration: the real train_one_epoch / checkpoint functions
# ---------------------------------------------------------------------------
class _Loader:
    """A handful of synthetic batches shaped like MixCollate's output."""

    def __init__(self, batches: int = 6, batch_size: int = 4, features: int = 4) -> None:
        torch.manual_seed(7)
        self._batches = [
            (
                torch.randn(batch_size, features),
                torch.randint(0, 2, (batch_size,)),
            )
            for _ in range(batches)
        ]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def _train_config(**overrides):
    from models.training.train_classifier import TrainConfig

    config = TrainConfig(
        epochs=1,
        mixed_precision=False,
        gradient_clip_norm=1.0,
        accumulation_steps=1,
        ema=True,
        ema_decay=0.9,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestTrainOneEpochDrivesTheAverage:
    def test_the_average_is_updated_once_per_optimiser_step(self) -> None:
        from models.training.train_classifier import train_one_epoch

        model = _model()
        ema = ModelEma(model, decay=0.9)
        loader = _Loader(batches=6)

        train_one_epoch(
            model,
            loader,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            scaler=None,
            scheduler=None,
            step_per_batch=False,
            config=_train_config(),
            device="cpu",
            epoch=1,
            ema=ema,
        )
        assert ema.steps == 6

    def test_gradient_accumulation_steps_the_average_less_often(self) -> None:
        """The average must follow optimiser steps, not batches."""
        from models.training.train_classifier import train_one_epoch

        model = _model()
        ema = ModelEma(model, decay=0.9)

        train_one_epoch(
            model,
            _Loader(batches=6),
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            scaler=None,
            scheduler=None,
            step_per_batch=False,
            config=_train_config(accumulation_steps=3),
            device="cpu",
            epoch=1,
            ema=ema,
        )
        assert ema.steps == 2

    def test_no_average_requested_means_no_average_kept(self) -> None:
        from models.training.train_classifier import train_one_epoch

        model = _model()
        train_one_epoch(
            model,
            _Loader(),
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            scaler=None,
            scheduler=None,
            step_per_batch=False,
            config=_train_config(ema=False),
            device="cpu",
            epoch=1,
            ema=None,
        )

    def test_the_average_diverges_from_the_weights_during_training(self) -> None:
        """If these stayed identical the averaging would be doing nothing."""
        from models.training.train_classifier import train_one_epoch

        model = _model()
        ema = ModelEma(model, decay=0.9)
        train_one_epoch(
            model,
            _Loader(batches=6),
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.5),
            scaler=None,
            scheduler=None,
            step_per_batch=False,
            config=_train_config(),
            device="cpu",
            epoch=1,
            ema=ema,
        )
        assert not torch.allclose(ema.module[0].weight, model[0].weight)


class TestCheckpointCarriesTheAverage:
    def test_a_resumed_run_keeps_the_smoothing(self, tmp_path) -> None:
        """The failure this guards against is silent: a resume without the
        average restarts it from the live weights, losing every step of
        smoothing while still reporting an EMA score."""
        from models.training.train_classifier import (
            TrainingHistory,
            load_checkpoint,
            save_checkpoint,
        )

        model = _model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema = ModelEma(model, decay=0.9)
        for _ in range(30):
            _step(model, 0.05)
            ema.update(model)

        path = tmp_path / "last.pt"
        config = _train_config()
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            ema=ema,
            epoch=3,
            best_top1=42.0,
            epochs_without_improvement=0,
            history=TrainingHistory(config={}),
            config=config,
            num_classes=2,
            class_names=["a", "b"],
            adapted=False,
        )

        fresh_model = _model(seed=9)
        fresh_ema = ModelEma(fresh_model, decay=0.9)
        next_epoch, best, _, _ = load_checkpoint(
            path,
            model=fresh_model,
            optimizer=torch.optim.SGD(fresh_model.parameters(), lr=0.1),
            scheduler=None,
            scaler=None,
            device="cpu",
            ema=fresh_ema,
        )

        assert next_epoch == 4
        assert best == pytest.approx(42.0)
        assert fresh_ema.steps == 30
        assert torch.allclose(fresh_ema.module[0].weight, ema.module[0].weight)

    def test_a_checkpoint_written_without_an_average_still_loads(self, tmp_path) -> None:
        """Backwards compatibility with every checkpoint written so far."""
        from models.training.train_classifier import (
            TrainingHistory,
            load_checkpoint,
            save_checkpoint,
        )

        model = _model()
        path = tmp_path / "last.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            scheduler=None,
            scaler=None,
            ema=None,
            epoch=1,
            best_top1=0.0,
            epochs_without_improvement=0,
            history=TrainingHistory(config={}),
            config=_train_config(ema=False),
            num_classes=2,
            class_names=["a", "b"],
            adapted=False,
        )

        fresh = _model(seed=3)
        ema = ModelEma(fresh, decay=0.9)
        load_checkpoint(
            path,
            model=fresh,
            optimizer=torch.optim.SGD(fresh.parameters(), lr=0.1),
            scheduler=None,
            scaler=None,
            device="cpu",
            ema=ema,
        )
        assert ema.steps == 0
