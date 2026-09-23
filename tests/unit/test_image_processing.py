"""Unit tests for image preprocessing.

Preprocessing bugs are silent: the API returns 200 and the predictions are
quietly wrong. These tests pin down the behaviour that keeps serving matched
to training.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from api.exceptions import InvalidImageError
from api.utils.image_processing import (
    CLASSIFICATION_PREPROCESS,
    DETECTION_PREPROCESS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessConfig,
    decode_image,
    image_hash,
    preprocess,
    preprocess_batch,
    resize_center_crop,
    resize_letterbox,
    scale_boxes_to_original,
    to_array,
)
from tests.conftest import make_image


class TestDecodeImage:
    def test_decodes_rgb(self, sample_image: bytes) -> None:
        img = decode_image(sample_image)
        assert img.mode == "RGB"
        assert img.size == (224, 224)

    def test_converts_grayscale_to_rgb(self, grayscale_image: bytes) -> None:
        """Models expect 3 channels; a 1-channel input would crash them."""
        assert decode_image(grayscale_image).mode == "RGB"

    def test_composites_alpha_onto_white(self) -> None:
        """Transparent regions become white, not black.

        Dropping the alpha channel instead of compositing leaves transparent
        pixels as black, which looks like a dark object to the model.
        """
        buffer = io.BytesIO()
        Image.new("RGBA", (64, 64), (255, 0, 0, 0)).save(buffer, "PNG")
        img = decode_image(buffer.getvalue())
        assert img.mode == "RGB"
        assert np.asarray(img)[0, 0].tolist() == [255, 255, 255]

    def test_applies_exif_orientation(self) -> None:
        """A photo tagged 'rotate 90' must arrive upright.

        Phone cameras record orientation in metadata rather than rotating the
        pixels. Ignoring it feeds the model a sideways image.
        """
        img = Image.new("RGB", (100, 50), (255, 0, 0))
        exif = img.getexif()
        exif[274] = 6  # Orientation tag: rotate 90 degrees clockwise
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", exif=exif)

        decoded = decode_image(buffer.getvalue())
        # 100x50 rotated 90 degrees becomes 50x100.
        assert decoded.size == (50, 100)

    def test_raises_on_garbage(self, not_an_image: bytes) -> None:
        with pytest.raises(InvalidImageError):
            decode_image(not_an_image)


class TestResize:
    def test_center_crop_produces_exact_size(self) -> None:
        img = Image.new("RGB", (640, 480))
        assert resize_center_crop(img, (224, 224)).size == (224, 224)

    def test_center_crop_handles_portrait(self) -> None:
        img = Image.new("RGB", (480, 640))
        assert resize_center_crop(img, (224, 224)).size == (224, 224)

    def test_letterbox_preserves_aspect_ratio(self) -> None:
        """A 2:1 image must not be squashed; it gets padded instead."""
        img = Image.new("RGB", (1280, 640), (255, 0, 0))
        out, scale, (pad_x, pad_y) = resize_letterbox(img, (640, 640))

        assert out.size == (640, 640)
        assert scale == pytest.approx(0.5)
        assert pad_x == pytest.approx(0.0)
        assert pad_y == pytest.approx(160.0)

    def test_letterbox_pads_with_grey(self) -> None:
        img = Image.new("RGB", (1280, 640), (255, 0, 0))
        out, _, _ = resize_letterbox(img, (640, 640), pad_value=114)
        # Top-left corner falls inside the padded band.
        assert np.asarray(out)[0, 0].tolist() == [114, 114, 114]

    def test_letterbox_square_image_needs_no_padding(self) -> None:
        img = Image.new("RGB", (800, 800))
        _, scale, (pad_x, pad_y) = resize_letterbox(img, (640, 640))
        assert scale == pytest.approx(0.8)
        assert (pad_x, pad_y) == (0.0, 0.0)


class TestToArray:
    def test_produces_chw_float32(self) -> None:
        arr = to_array(Image.new("RGB", (224, 224)), CLASSIFICATION_PREPROCESS)
        assert arr.shape == (3, 224, 224)
        assert arr.dtype == np.float32

    def test_applies_imagenet_normalisation(self) -> None:
        """A mid-grey pixel must map to the expected normalised value.

        This is the check that catches a normalisation mismatch between
        training and serving, which silently costs accuracy.
        """
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        arr = to_array(img, PreprocessConfig(size=(32, 32)))
        expected_r = (128 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        assert arr[0, 0, 0] == pytest.approx(expected_r, abs=1e-5)

    def test_skips_normalisation_when_disabled(self) -> None:
        """YOLO wants plain 0-1 input, not ImageNet-normalised."""
        img = Image.new("RGB", (32, 32), (255, 255, 255))
        arr = to_array(img, PreprocessConfig(size=(32, 32), normalize=False))
        assert arr.max() == pytest.approx(1.0)
        assert arr.min() >= 0.0

    def test_channels_last_when_requested(self) -> None:
        arr = to_array(
            Image.new("RGB", (64, 64)), PreprocessConfig(size=(64, 64), channels_first=False)
        )
        assert arr.shape == (64, 64, 3)


class TestPreprocess:
    def test_adds_batch_dimension(self, sample_image: bytes) -> None:
        assert preprocess(sample_image, CLASSIFICATION_PREPROCESS).array.shape == (1, 3, 224, 224)

    def test_detection_shape(self, sample_image: bytes) -> None:
        assert preprocess(sample_image, DETECTION_PREPROCESS).array.shape == (1, 3, 640, 640)

    def test_records_original_size(self) -> None:
        result = preprocess(make_image(800, 600), CLASSIFICATION_PREPROCESS)
        assert result.original_size == (800, 600)

    def test_warns_when_resized(self) -> None:
        result = preprocess(make_image(800, 600), CLASSIFICATION_PREPROCESS)
        assert any("resized" in w for w in result.warnings)

    def test_no_warning_when_already_correct_size(self, sample_image: bytes) -> None:
        assert preprocess(sample_image, CLASSIFICATION_PREPROCESS).warnings == []

    def test_rejects_unknown_resize_mode(self, sample_image: bytes) -> None:
        with pytest.raises(ValueError, match="unknown resize_mode"):
            preprocess(sample_image, PreprocessConfig(resize_mode="teleport"))

    def test_batch_stacks_correctly(self, sample_image: bytes) -> None:
        stacked, results = preprocess_batch([sample_image] * 5, CLASSIFICATION_PREPROCESS)
        assert stacked.shape == (5, 3, 224, 224)
        assert len(results) == 5


class TestScaleBoxesToOriginal:
    """Box un-letterboxing. Getting this wrong misplaces every box."""

    def test_round_trip(self) -> None:
        """Map a box forward through letterboxing, then back, and recover it."""
        result = preprocess(make_image(1920, 1080), DETECTION_PREPROCESS)

        original = np.array([[200.0, 300.0, 800.0, 700.0]])
        # Forward: original -> letterboxed space.
        forward = original.copy()
        forward[:, [0, 2]] = forward[:, [0, 2]] * result.scale + result.pad[0]
        forward[:, [1, 3]] = forward[:, [1, 3]] * result.scale + result.pad[1]

        recovered = scale_boxes_to_original(forward, result)
        np.testing.assert_allclose(recovered, original, atol=0.01)

    def test_clips_to_image_bounds(self) -> None:
        """A box predicted past the edge is clipped, not returned as negative."""
        result = preprocess(make_image(640, 480), DETECTION_PREPROCESS)
        boxes = np.array([[-500.0, -500.0, 5000.0, 5000.0]])
        out = scale_boxes_to_original(boxes, result)

        assert out[0, 0] >= 0 and out[0, 1] >= 0
        assert out[0, 2] <= 640 and out[0, 3] <= 480

    def test_empty_input(self) -> None:
        result = preprocess(make_image(640, 480), DETECTION_PREPROCESS)
        assert scale_boxes_to_original(np.empty((0, 4)), result).size == 0


class TestImageHash:
    def test_same_bytes_same_hash(self, sample_image: bytes) -> None:
        assert image_hash(sample_image) == image_hash(sample_image)

    def test_different_bytes_different_hash(self, sample_image: bytes, sample_jpeg: bytes) -> None:
        assert image_hash(sample_image) != image_hash(sample_jpeg)

    def test_is_hex_sha256(self, sample_image: bytes) -> None:
        digest = image_hash(sample_image)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
