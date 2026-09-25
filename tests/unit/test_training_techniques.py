"""Two properties of the fine-tuning loop that fail silently.

The brief asks for mixed precision, gradient clipping and learning-rate
scheduling. All three were active in the run that produced 78.91%, and the
training history is the evidence: the GradScaler loss scale halved five times
on overflow, per-epoch gradient norms sat between 1.36 and 3.10 against a 1.0
threshold, and the rate warmed up then annealed 300x.

Most of that is PyTorch's code. `autocast`, `GradScaler` and
`clip_grad_norm_` are library functions with their own test suites, and
wrapping them in assertions here would test PyTorch rather than this project.

Two things are ours, and both are wrong in a way nothing reports:

1. **`step_per_batch`.** Cosine is a per-batch schedule. Return False and it
   steps once per epoch instead, decaying the rate twenty times too slowly.
   Training completes, the loss falls, the model is just worse.

2. **`unscale_` before `clip_grad_norm_`.** Under AMP the loss is multiplied
   by a large scale factor before `backward`, so gradients carry that factor.
   Clipping them first caps an inflated norm, which makes the configured
   `max_norm=1.0` really `1.0 * loss_scale` - a threshold that moves every
   time the scaler adapts.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap

import pytest

torch = pytest.importorskip("torch")

from models.training.train_classifier import TrainConfig, build_scheduler


def _optimizer(lr: float = 3e-4):
    model = torch.nn.Linear(4, 2)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _trace(scheduler, optimizer, steps: int) -> list[float]:
    """The rate the optimiser actually sees, step by step."""
    seen = []
    for _ in range(steps):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return seen


class TestSchedulerSteppingGranularity:
    """The flag that decays the rate at the wrong rate when it is wrong."""

    @pytest.mark.parametrize(
        ("name", "per_batch"),
        [("cosine", True), ("onecycle", True), ("step", False), ("plateau", False)],
    )
    def test_each_schedule_reports_how_it_should_be_stepped(
        self, name: str, per_batch: bool
    ) -> None:
        optimizer = _optimizer()
        config = TrainConfig(scheduler=name, epochs=6, learning_rate=1e-3)
        _, step_per_batch = build_scheduler(optimizer, config, steps_per_epoch=10)
        assert step_per_batch is per_batch

    def test_an_unknown_scheduler_is_refused(self) -> None:
        """Rather than silently falling through to a constant rate."""
        with pytest.raises(ValueError, match="unknown scheduler"):
            build_scheduler(_optimizer(), TrainConfig(scheduler="nope"), steps_per_epoch=10)


class TestCosineScheduleShape:
    """The lambda is ours, so its shape is worth pinning: warmup, peak, anneal."""

    def setup_method(self) -> None:
        self.config = TrainConfig(
            scheduler="cosine",
            epochs=10,
            learning_rate=3e-4,
            min_learning_rate=1e-6,
            warmup_ratio=0.1,
        )
        self.steps = 20 * self.config.epochs

    def _rates(self) -> list[float]:
        optimizer = _optimizer()
        scheduler, _ = build_scheduler(optimizer, self.config, steps_per_epoch=20)
        return _trace(scheduler, optimizer, self.steps)

    def test_it_warms_up_then_anneals(self) -> None:
        rates = self._rates()
        peak = max(rates)
        assert rates[0] < peak, "no warmup: it starts at the peak"
        assert rates[-1] < 0.02 * peak, f"barely annealed: ended at {rates[-1]:.2e}"

    def test_it_reaches_the_configured_rate_and_no_higher(self) -> None:
        rates = self._rates()
        assert max(rates) == pytest.approx(self.config.learning_rate, rel=1e-3)

    def test_the_decay_is_monotonic_and_bounded_below(self) -> None:
        rates = self._rates()
        decay = rates[rates.index(max(rates)) :]
        assert all(a >= b for a, b in itertools.pairwise(decay))
        assert min(decay) >= self.config.min_learning_rate * 0.99


class TestUnscaleHappensBeforeClipping:
    """Checked in the source, because the branch needs a GPU.

    The AMP path runs only when `device == "cuda"`, and a CPU-only torch
    raises on the first `.to("cuda")`, so there is nothing to execute on a
    runner without one. The property is a statement ordering, and the AST
    states it exactly.
    """

    @staticmethod
    def _calls_in_step_block() -> list[str]:
        """Function names called inside `if is_step:`, in source order."""
        import models.training.train_classifier as trainer

        tree = ast.parse(textwrap.dedent(inspect.getsource(trainer.train_one_epoch)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "is_step"
            ):
                # Sorted by source position: ast.walk is breadth-first, so
                # its order has nothing to do with the order things run in.
                found = []
                for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                    func = call.func
                    name = (
                        func.attr
                        if isinstance(func, ast.Attribute)
                        else func.id if isinstance(func, ast.Name) else None
                    )
                    if name:
                        found.append((call.lineno, call.col_offset, name))
                return [name for _, _, name in sorted(found)]
        raise AssertionError("no `if is_step:` block found in train_one_epoch")

    def test_the_step_block_unscales_clips_and_steps(self) -> None:
        calls = self._calls_in_step_block()
        for expected in ("unscale_", "clip_grad_norm_", "step", "zero_grad"):
            assert expected in calls, f"{expected} missing from the step block: {calls}"

    def test_unscaling_precedes_clipping(self) -> None:
        calls = self._calls_in_step_block()
        assert calls.index("unscale_") < calls.index("clip_grad_norm_"), (
            f"clipping runs before unscaling, so max_norm scales with the loss "
            f"scale instead of meaning 1.0: {calls}"
        )

    def test_clipping_precedes_the_optimiser_step(self) -> None:
        """Clipping after the step would not affect the update at all."""
        calls = self._calls_in_step_block()
        assert calls.index("clip_grad_norm_") < calls.index("step"), calls
