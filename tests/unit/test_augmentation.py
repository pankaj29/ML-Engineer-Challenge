"""Unit tests for the custom augmentation pipeline.

Augmentation is the part of training whose bugs never announce themselves.
Every operation here returns a valid image and every mixed batch trains to
completion, so a sign error in `shear_x`, a magnitude scaled the wrong way, or
a `lam` that does not match the area actually pasted all produce a model that
is simply a bit worse, with nothing in the logs to say why.

So these tests check the arithmetic rather than the absence of exceptions:

* Each operation is a no-op at magnitude 0 and visibly changes the image at
  magnitude 1. An operation wired to the wrong PIL call usually fails one of
  those two.
* CutMix's `lam` is compared against the pasted area counted from the pixels,
  because the whole correctness of the loss rests on those agreeing.
* `EvalTransform` is deterministic and `TrainTransform` is not, which is the
  one property that makes validation numbers comparable between epochs.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from PIL import Image

from models.training.augmentation import (
    RANDAUGMENT_OPS,
    AugmentationConfig,
    EvalTransform,
    MixCollate,
    RandAugment,
    RandomErasing,
    RandomResizedCropCustom,
    TrainTransform,
    cutmix_batch,
    mix_criterion,
    mixup_batch,
)


@pytest.fixture(autouse=True)
def fixed_seed():
    """Randomised augmentation needs a fixed seed to be assertable."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)


def photo(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """A structured image. Flat colour would hide most operations."""
    rng = np.random.default_rng(7)
    array = rng.integers(40, 215, (size[1], size[0], 3), dtype=np.uint8)
    array[: size[1] // 2, :, 0] = 240  # a bright band, so rotation and shear show
    array[:, : size[0] // 2, 2] = 20  # a dark column, so flips show
    return Image.fromarray(array)


def pixels(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.int16)


def differs(a: Image.Image, b: Image.Image) -> bool:
    return not np.array_equal(pixels(a), pixels(b))


# ---------------------------------------------------------------------------
# The 13 RandAugment operations
# ---------------------------------------------------------------------------
class TestOperations:
    def test_there_are_thirteen(self) -> None:
        assert len(RANDAUGMENT_OPS) == 13

    def test_invert_is_excluded(self) -> None:
        """A judgement call worth pinning: it destroys natural photos."""
        assert "invert" not in {op.__name__ for op in RANDAUGMENT_OPS}

    @pytest.mark.parametrize("op", RANDAUGMENT_OPS, ids=lambda op: op.__name__)
    def test_returns_an_image_of_the_same_size_and_mode(self, op) -> None:
        source = photo()
        result = op(source, 0.5)
        assert isinstance(result, Image.Image)
        assert result.size == source.size
        assert result.mode == source.mode

    @pytest.mark.parametrize("op", RANDAUGMENT_OPS, ids=lambda op: op.__name__)
    def test_magnitude_one_changes_the_image(self, op) -> None:
        """An operation wired to the wrong call often silently does nothing."""
        source = photo()
        assert differs(source, op(source.copy(), 1.0)), f"{op.__name__} had no effect"

    @pytest.mark.parametrize(
        "op",
        [o for o in RANDAUGMENT_OPS if o.__name__.startswith(("adjust_", "shear", "translate"))],
        ids=lambda op: op.__name__,
    )
    def test_magnitude_zero_is_a_no_op(self, op) -> None:
        """The operations with a continuous scale must vanish at zero.

        autocontrast, equalize, posterize and solarize are excluded: they are
        not parameterised this way and change the image at any magnitude.
        """
        source = photo()
        assert not differs(source, op(source.copy(), 0.0)), f"{op.__name__} changed at magnitude 0"

    @pytest.mark.parametrize("op", RANDAUGMENT_OPS, ids=lambda op: op.__name__)
    def test_pixels_stay_in_range(self, op) -> None:
        values = pixels(op(photo(), 1.0))
        assert values.min() >= 0
        assert values.max() <= 255

    def test_shear_x_and_shear_y_differ(self) -> None:
        """A copy-paste between the two axes is an easy and silent mistake."""
        from models.training.augmentation import shear_x, shear_y

        source = photo()
        assert differs(shear_x(source.copy(), 0.8), shear_y(source.copy(), 0.8))

    def test_translate_x_and_translate_y_differ(self) -> None:
        from models.training.augmentation import translate_x, translate_y

        source = photo()
        assert differs(translate_x(source.copy(), 0.8), translate_y(source.copy(), 0.8))

    def test_brightness_increases_mean_intensity(self) -> None:
        """A sign error here would darken instead of brighten."""
        from models.training.augmentation import adjust_brightness

        source = photo()
        assert pixels(adjust_brightness(source.copy(), 1.0)).mean() > pixels(source).mean()

    def test_posterize_reduces_distinct_values(self) -> None:
        from models.training.augmentation import posterize

        source = photo()
        assert len(np.unique(pixels(posterize(source.copy(), 1.0)))) < len(
            np.unique(pixels(source))
        )


class TestRandAugment:
    def test_applies_the_requested_number_of_operations(self, monkeypatch) -> None:
        applied: list[str] = []
        original = random.sample

        def spy(population, k):
            chosen = original(population, k)
            applied.extend(op.__name__ for op in chosen)
            return chosen

        monkeypatch.setattr(random, "sample", spy)
        RandAugment(num_ops=3, magnitude=0.5)(photo())
        assert len(applied) == 3

    def test_never_asks_for_more_operations_than_exist(self) -> None:
        assert RandAugment(num_ops=99, magnitude=0.5)(photo()) is not None

    def test_it_changes_the_image(self) -> None:
        source = photo()
        assert differs(source, RandAugment(num_ops=2, magnitude=0.9)(source.copy()))

    def test_repr_names_its_settings(self) -> None:
        assert "num_ops=2" in repr(RandAugment(num_ops=2, magnitude=0.4))


# ---------------------------------------------------------------------------
# Geometric and occlusion
# ---------------------------------------------------------------------------
class TestRandomResizedCrop:
    def test_output_is_the_requested_size(self) -> None:
        cropped = RandomResizedCropCustom(size=(32, 32))(photo((80, 60)))
        assert cropped.size == (32, 32)

    def test_a_non_square_source_still_gives_a_square_output(self) -> None:
        assert RandomResizedCropCustom(size=(48, 48))(photo((200, 50))).size == (48, 48)

    def test_an_extreme_aspect_ratio_falls_back_to_a_centre_crop(self) -> None:
        """Ten failed attempts must not loop forever."""
        cropper = RandomResizedCropCustom(size=(32, 32), scale=(0.99, 1.0), ratio=(10.0, 10.0))
        assert cropper(photo((64, 64))).size == (32, 32)

    def test_successive_crops_differ(self) -> None:
        cropper = RandomResizedCropCustom(size=(32, 32))
        source = photo((128, 128))
        assert differs(cropper(source.copy()), cropper(source.copy()))

    def test_the_default_scale_floor_is_gentler_than_torchvision(self) -> None:
        """0.35, not 0.08: on a 64px image, 8% of the area is 18x18 pixels,
        which often contains no subject while keeping the original label."""
        assert RandomResizedCropCustom().scale[0] == pytest.approx(0.35)


class TestRandomErasing:
    def test_probability_zero_never_erases(self) -> None:
        tensor = torch.ones(3, 32, 32)
        assert torch.equal(RandomErasing(probability=0.0)(tensor.clone()), tensor)

    def test_probability_one_erases(self) -> None:
        tensor = torch.ones(3, 32, 32)
        assert not torch.equal(RandomErasing(probability=1.0)(tensor.clone()), tensor)

    def test_it_erases_a_rectangle_not_the_whole_image(self) -> None:
        tensor = torch.ones(3, 64, 64)
        erased = RandomErasing(probability=1.0, scale=(0.02, 0.2))(tensor.clone())
        untouched = (erased == 1.0).all(dim=0)
        assert untouched.any(), "the whole image was erased"
        assert not untouched.all(), "nothing was erased"

    def test_the_erased_area_is_within_the_configured_range(self) -> None:
        tensor = torch.ones(3, 64, 64)
        erased = RandomErasing(probability=1.0, scale=(0.05, 0.2))(tensor.clone())
        changed = (erased != 1.0).any(dim=0).float().mean().item()
        assert 0.01 <= changed <= 0.35, f"erased {changed:.1%} of the image"

    def test_shape_and_dtype_survive(self) -> None:
        tensor = torch.ones(3, 32, 32)
        result = RandomErasing(probability=1.0)(tensor.clone())
        assert result.shape == tensor.shape
        assert result.dtype == tensor.dtype


# ---------------------------------------------------------------------------
# MixUp and CutMix
# ---------------------------------------------------------------------------
class TestMixUp:
    def test_returns_the_four_tuple_the_training_loop_expects(self) -> None:
        images = torch.rand(8, 3, 16, 16)
        targets = torch.arange(8)
        mixed, a, b, lam = mixup_batch(images, targets, alpha=0.2)

        assert mixed.shape == images.shape
        assert a.shape == targets.shape == b.shape
        assert 0.0 <= lam <= 1.0

    def test_targets_a_is_the_original_order(self) -> None:
        targets = torch.arange(8)
        _, a, _, _ = mixup_batch(torch.rand(8, 3, 16, 16), targets, alpha=0.2)
        assert torch.equal(a, targets)

    def test_targets_b_is_a_permutation(self) -> None:
        targets = torch.arange(8)
        _, _, b, _ = mixup_batch(torch.rand(8, 3, 16, 16), targets, alpha=0.2)
        assert sorted(b.tolist()) == sorted(targets.tolist())

    def test_the_blend_matches_lam(self) -> None:
        """The arithmetic the loss weighting depends on."""
        images = torch.rand(6, 3, 8, 8)
        targets = torch.arange(6)
        mixed, _, _, _lam = mixup_batch(images, targets, alpha=0.2)

        # mixed = lam*images + (1-lam)*images[perm], so the batch mean is
        # preserved whatever the permutation was.
        assert mixed.mean().item() == pytest.approx(images.mean().item(), abs=1e-5)

    def test_alpha_zero_disables_mixing(self) -> None:
        images = torch.rand(4, 3, 8, 8)
        mixed, _, _, lam = mixup_batch(images, torch.arange(4), alpha=0.0)
        assert lam == 1.0
        assert torch.allclose(mixed, images)


class TestCutMix:
    def test_returns_the_four_tuple(self) -> None:
        images = torch.rand(8, 3, 32, 32)
        targets = torch.arange(8)
        mixed, a, _b, lam = cutmix_batch(images, targets, alpha=1.0)

        assert mixed.shape == images.shape
        assert torch.equal(a, targets)
        assert 0.0 <= lam <= 1.0

    def test_lam_equals_the_area_actually_left_untouched(self, monkeypatch) -> None:
        """The correctness claim the implementation makes.

        A box clipped at the image edge covers less than the Beta draw
        intended, so `lam` is recomputed from the clipped box. If it were not,
        the loss would weight the two labels wrongly on every clipped sample,
        quietly and forever.

        The permutation is pinned to a swap. Left to chance, `randperm` on a
        two-image batch returns the identity half the time, and an image
        pasted onto itself is indistinguishable from no paste at all.
        """
        monkeypatch.setattr(torch, "randperm", lambda n, **kw: torch.tensor([1, 0]))

        for _ in range(25):
            images = torch.zeros(2, 1, 40, 40)
            images[1] = 1.0  # so pasted pixels are identifiable
            mixed, _, _, lam = cutmix_batch(images, torch.tensor([0, 1]), alpha=1.0)

            pasted = (mixed[0] == 1.0).float().mean().item()
            assert lam == pytest.approx(
                1.0 - pasted, abs=1e-6
            ), f"lam {lam:.4f} but {pasted:.4f} of the image was pasted"

    def test_it_pastes_a_contiguous_rectangle(self, monkeypatch) -> None:
        monkeypatch.setattr(torch, "randperm", lambda n, **kw: torch.tensor([1, 0]))
        images = torch.zeros(2, 1, 32, 32)
        images[1] = 1.0
        mixed, _, _, lam = cutmix_batch(images, torch.tensor([0, 1]), alpha=1.0)

        if lam < 1.0:  # something was pasted
            rows = (mixed[0, 0] == 1.0).any(dim=1).nonzero().flatten()
            cols = (mixed[0, 0] == 1.0).any(dim=0).nonzero().flatten()
            assert rows.max() - rows.min() + 1 == len(rows), "pasted rows are not contiguous"
            assert cols.max() - cols.min() + 1 == len(cols), "pasted columns are not contiguous"

    def test_the_original_batch_is_not_modified(self) -> None:
        """It clones; without that, the source batch would be corrupted."""
        images = torch.rand(4, 3, 16, 16)
        before = images.clone()
        cutmix_batch(images, torch.arange(4), alpha=1.0)
        assert torch.equal(images, before)

    def test_alpha_zero_disables_mixing(self) -> None:
        images = torch.rand(4, 3, 16, 16)
        _, _, _, lam = cutmix_batch(images, torch.arange(4), alpha=0.0)
        assert lam == pytest.approx(1.0)


class TestMixCriterion:
    def test_lam_one_uses_only_the_first_target(self) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        predictions = torch.randn(4, 10)
        a, b = torch.arange(4), torch.arange(4).flip(0)

        assert mix_criterion(criterion, predictions, a, b, 1.0).item() == pytest.approx(
            criterion(predictions, a).item()
        )

    def test_it_is_the_weighted_sum(self) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        predictions = torch.randn(4, 10)
        a, b = torch.arange(4), torch.arange(4).flip(0)
        lam = 0.3

        expected = lam * criterion(predictions, a) + (1 - lam) * criterion(predictions, b)
        assert mix_criterion(criterion, predictions, a, b, lam).item() == pytest.approx(
            expected.item(), abs=1e-6
        )

    def test_the_loss_stays_finite(self) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        loss = mix_criterion(criterion, torch.randn(4, 10), torch.arange(4), torch.arange(4), 0.5)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# The collate function
# ---------------------------------------------------------------------------
class TestMixCollate:
    @staticmethod
    def _batch(n: int = 8):
        return [(torch.rand(3, 16, 16), i % 4) for i in range(n)]

    def test_always_returns_four_values(self) -> None:
        """The training loop unpacks four; anything else is a TypeError."""
        images, a, b, lam = MixCollate(AugmentationConfig(mix_prob=1.0))(self._batch())
        assert images.shape[0] == 8
        assert len(a) == len(b) == 8
        assert isinstance(lam, float)

    def test_mix_prob_zero_passes_the_batch_through(self) -> None:
        collate = MixCollate(AugmentationConfig(mix_prob=0.0))
        _images, a, b, lam = collate(self._batch())
        assert lam == 1.0
        assert torch.equal(a, b), "labels were mixed when mixing was disabled"

    def test_a_single_image_batch_is_never_mixed(self) -> None:
        """Mixing an image with itself is pointless and would skew lam."""
        _, a, b, lam = MixCollate(AugmentationConfig(mix_prob=1.0))(self._batch(1))
        assert lam == 1.0
        assert torch.equal(a, b)

    def test_both_alphas_zero_disables_mixing(self) -> None:
        config = AugmentationConfig(mix_prob=1.0, mixup_alpha=0.0, cutmix_alpha=0.0)
        _, a, b, lam = MixCollate(config)(self._batch())
        assert lam == 1.0
        assert torch.equal(a, b)

    def test_cutmix_only_configuration(self) -> None:
        config = AugmentationConfig(mix_prob=1.0, mixup_alpha=0.0, cutmix_alpha=1.0)
        images, _, _, _ = MixCollate(config)(self._batch())
        assert images.shape == (8, 3, 16, 16)

    def test_mixup_only_configuration(self) -> None:
        config = AugmentationConfig(mix_prob=1.0, mixup_alpha=0.2, cutmix_alpha=0.0)
        images, _, _, _ = MixCollate(config)(self._batch())
        assert images.shape == (8, 3, 16, 16)


# ---------------------------------------------------------------------------
# Composed pipelines
# ---------------------------------------------------------------------------
class TestTrainTransform:
    def test_produces_a_normalised_tensor_of_the_right_shape(self) -> None:
        result = TrainTransform(AugmentationConfig(image_size=32))(photo())
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)
        assert result.dtype == torch.float32

    def test_normalisation_moves_the_data_off_the_zero_to_one_range(self) -> None:
        result = TrainTransform(AugmentationConfig(image_size=32))(photo())
        assert result.min() < 0.0, "does not look normalised"

    def test_it_is_random(self) -> None:
        """Two runs over the same image must differ, or it is not augmenting."""
        transform = TrainTransform(AugmentationConfig(image_size=32))
        source = photo()
        assert not torch.equal(transform(source.copy()), transform(source.copy()))

    def test_every_stage_can_be_turned_off(self) -> None:
        config = AugmentationConfig(
            image_size=32,
            random_resized_crop=False,
            horizontal_flip_prob=0.0,
            randaugment=False,
            random_erasing_prob=0.0,
        )
        assert TrainTransform(config)(photo()).shape == (3, 32, 32)

    def test_a_greyscale_image_is_handled(self) -> None:
        grey = photo().convert("L")
        assert TrainTransform(AugmentationConfig(image_size=32))(grey).shape == (3, 32, 32)


class TestEvalTransform:
    def test_is_deterministic(self) -> None:
        """Validation numbers are only comparable between epochs if this holds."""
        transform = EvalTransform(AugmentationConfig(image_size=32))
        source = photo()
        assert torch.equal(transform(source.copy()), transform(source.copy()))

    def test_produces_the_same_shape_as_the_training_transform(self) -> None:
        config = AugmentationConfig(image_size=32)
        assert EvalTransform(config)(photo()).shape == TrainTransform(config)(photo()).shape

    def test_it_does_not_augment(self) -> None:
        """Ten runs, all identical: no randomness anywhere in the path."""
        transform = EvalTransform(AugmentationConfig(image_size=32))
        source = photo()
        first = transform(source.copy())
        assert all(torch.equal(first, transform(source.copy())) for _ in range(10))


class TestAugmentationConfig:
    def test_describe_is_json_serialisable(self) -> None:
        """It is written into the training history for reproducibility."""
        import json

        json.dumps(AugmentationConfig().describe())

    def test_describe_reports_the_randaugment_setting(self) -> None:
        described = AugmentationConfig(randaugment_ops=3, randaugment_magnitude=0.5).describe()
        assert described["randaugment"] == "3 ops @ magnitude 0.5"

    def test_describe_says_off_when_randaugment_is_disabled(self) -> None:
        assert AugmentationConfig(randaugment=False).describe()["randaugment"] == "off"

    def test_vertical_flip_is_off_by_default(self) -> None:
        """Most photos have a consistent 'up'; teaching otherwise costs accuracy."""
        assert AugmentationConfig().vertical_flip_prob == 0.0

    def test_horizontal_flip_is_on_by_default(self) -> None:
        assert AugmentationConfig().horizontal_flip_prob == 0.5

    def test_vertical_flip_can_be_enabled(self) -> None:
        """Off by default, but aerial and microscopy imagery wants it."""
        config = AugmentationConfig(
            image_size=32,
            vertical_flip_prob=1.0,
            horizontal_flip_prob=0.0,
            random_resized_crop=False,
            randaugment=False,
            random_erasing_prob=0.0,
        )
        flipped = TrainTransform(config)(photo((32, 32)))
        upright = EvalTransform(AugmentationConfig(image_size=32, random_erasing_prob=0.0))(
            photo((32, 32))
        )
        assert not torch.equal(flipped, upright)


class TestInvert:
    """Not in the RandAugment menu, but exported and worth keeping correct."""

    def test_it_inverts(self) -> None:
        from models.training.augmentation import invert

        source = photo()
        inverted = invert(source.copy(), 1.0)
        assert pixels(inverted).mean() == pytest.approx(255 - pixels(source).mean(), abs=1.0)

    def test_inverting_twice_restores_the_original(self) -> None:
        from models.training.augmentation import invert

        source = photo()
        assert not differs(source, invert(invert(source.copy(), 1.0), 1.0))
