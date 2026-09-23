"""Serving preprocessing must match the transform each model was evaluated with.

This is the highest-value test in the suite for its size, because the bug it
guards against is completely silent. Nothing crashes. No error is logged. The
API returns a confident answer. Accuracy is just quietly worse in production
than it was in validation, and there is no signal pointing at the cause.

It was written after finding a real instance: ``TINY_IMAGENET_PREPROCESS`` was
configured with ``resize_mode="center_crop", crop_pct=0.875``, while the
training pipeline's ``EvalTransform`` did a direct resize to 64x64. Serving
would have upscaled every image to 73px and cropped back to 64, throwing away
the border. Measured difference on identical input: **4.28** in normalised
units, on data whose standard deviation is 1.

The rule these tests encode is *not* "always centre-crop" or "never
centre-crop". It is: **match how the model was evaluated.**

* ImageNet weights were evaluated with resize-256 then centre-crop-224, so the
  ImageNet config does exactly that.
* Our Tiny-ImageNet model was validated with a direct resize, so its config
  does exactly that.

Getting this wrong in either direction costs accuracy.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from api.utils.image_processing import (
    CLASSIFICATION_PREPROCESS,
    DETECTION_PREPROCESS,
    TINY_IMAGENET_PREPROCESS,
    preprocess,
)
from models.training.augmentation import AugmentationConfig, EvalTransform


def make_image(width: int, height: int, seed: int = 0) -> bytes:
    """A noise image. Noise, not flat colour: a flat image survives almost any
    resize unchanged and would hide exactly the bug being tested for."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class TestTinyImageNetParity:
    """The serving config must reproduce EvalTransform bit for bit."""

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (64, 64),  # the native size - the common case
            (100, 60),  # landscape, needs resizing
            (60, 100),  # portrait
            (256, 256),  # larger square
            (512, 128),  # extreme aspect ratio
        ],
    )
    def test_matches_training_eval_transform(self, width: int, height: int) -> None:
        config = AugmentationConfig(image_size=64)
        data = make_image(width, height)

        expected = EvalTransform(config)(Image.open(io.BytesIO(data)).convert("RGB")).numpy()
        actual = preprocess(data, TINY_IMAGENET_PREPROCESS).array[0]

        assert actual.shape == expected.shape
        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-5,
            err_msg=(
                f"Serving preprocessing diverged from the training EvalTransform on a "
                f"{width}x{height} image. The model will score worse in production than "
                "in validation, with no error to indicate why. Check resize_mode, "
                "crop_pct, mean and std in TINY_IMAGENET_PREPROCESS."
            ),
        )

    def test_normalisation_constants_match(self) -> None:
        """Mean/std drift is the other half of this bug class, and is even
        harder to spot than a resize difference."""
        config = AugmentationConfig(image_size=64)
        assert TINY_IMAGENET_PREPROCESS.mean == config.mean
        assert TINY_IMAGENET_PREPROCESS.std == config.std

    def test_output_size_matches(self) -> None:
        config = AugmentationConfig(image_size=64)
        assert TINY_IMAGENET_PREPROCESS.size == (config.image_size, config.image_size)

    def test_uses_direct_resize_not_crop(self) -> None:
        """Explicit, so the intent survives a future refactor.

        A centre crop here would silently discard the image border.
        """
        assert TINY_IMAGENET_PREPROCESS.resize_mode == "stretch", (
            "Tiny-ImageNet serving must use a direct resize, matching EvalTransform. "
            "A centre crop upscales then crops, discarding the border."
        )


class TestImageNetConfigIsDeliberatelyDifferent:
    """The ImageNet config SHOULD centre-crop - that is how its weights were
    evaluated. These tests exist so nobody 'fixes' it to match the other one."""

    def test_uses_centre_crop(self) -> None:
        assert CLASSIFICATION_PREPROCESS.resize_mode == "center_crop"
        assert CLASSIFICATION_PREPROCESS.crop_pct == pytest.approx(0.875)

    def test_uses_imagenet_statistics(self) -> None:
        from api.utils.image_processing import IMAGENET_MEAN, IMAGENET_STD

        assert CLASSIFICATION_PREPROCESS.mean == IMAGENET_MEAN
        assert CLASSIFICATION_PREPROCESS.std == IMAGENET_STD

    def test_224_input(self) -> None:
        assert CLASSIFICATION_PREPROCESS.size == (224, 224)


class TestDetectionConfig:
    """YOLO has its own requirements, different again from both classifiers."""

    def test_letterboxes_rather_than_cropping(self) -> None:
        """Cropping a detection input would cut objects out of the frame, and
        squashing would distort every predicted box."""
        assert DETECTION_PREPROCESS.resize_mode == "letterbox"

    def test_does_not_normalise(self) -> None:
        """YOLO normalises internally and expects plain 0-1 input. Applying
        ImageNet mean/std here - the reflex from the classification path -
        silently wrecks the output."""
        assert DETECTION_PREPROCESS.normalize is False

    def test_640_input(self) -> None:
        assert DETECTION_PREPROCESS.size == (640, 640)


class TestRegistryPresetsResolve:
    """Every preset a registry entry can name must actually exist.

    A typo'd preprocess name falls back to the ImageNet default, which means a
    64px model would be fed 224px centre-cropped input. Another silent one.
    """

    def test_all_presets_present(self) -> None:
        from api.services.model_service import PREPROCESS_PRESETS

        assert set(PREPROCESS_PRESETS) >= {
            "imagenet_224",
            "tiny_imagenet_64",
            "yolo_640",
            "clip_224",
        }

    def test_tiny_imagenet_preset_is_the_fixed_one(self) -> None:
        from api.services.model_service import PREPROCESS_PRESETS

        preset = PREPROCESS_PRESETS["tiny_imagenet_64"]
        assert preset.size == (64, 64)
        assert preset.resize_mode == "stretch"

    def test_registered_models_name_a_real_preset(self) -> None:
        """Catches a registry entry pointing at a preset that does not exist."""
        import json
        from pathlib import Path

        from api.services.model_service import PREPROCESS_PRESETS

        registry = Path(__file__).resolve().parents[2] / "models" / "registry.json"
        if not registry.exists():
            pytest.skip("no registry; run scripts/prepare_models.py")

        for entry in json.loads(registry.read_text(encoding="utf-8")).get("models", []):
            name = entry.get("preprocess", "imagenet_224")
            assert name in PREPROCESS_PRESETS, (
                f"{entry['name']}:{entry['version']} names preprocess '{name}', which is "
                "not a registered preset. It would silently fall back to imagenet_224."
            )
