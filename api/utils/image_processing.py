"""Image decoding and preprocessing.

Plain English:
    A model does not accept a JPEG. It accepts a precisely shaped grid of
    numbers. This module is the bridge: bytes in, correctly shaped and scaled
    NumPy array out.

Getting this wrong is the single most common cause of "the model works in the
notebook but returns nonsense in production". The three classic mistakes, all
of which are handled explicitly below:

* **Wrong normalisation.** Training used ImageNet mean/std; serving used a
  plain 0-1 scale. Accuracy quietly collapses. We keep normalisation constants
  attached to each preprocessing config so they travel with the model.
* **Wrong channel order.** PIL gives RGB, OpenCV gives BGR. We standardise on
  RGB everywhere and convert exactly once, here.
* **Wrong resize semantics.** Classifiers expect a centre crop; detectors
  expect aspect-preserving letterbox padding, because squashing an image
  distorts the boxes. Both are implemented separately.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from api.exceptions import InvalidImageError
from api.logging_config import get_logger

logger = get_logger(__name__)

# Standard ImageNet statistics. Every torchvision/timm classification
# checkpoint was trained with these, so serving must reuse them exactly.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tiny-ImageNet's own channel statistics, computed over its training split.
# Used by models we fine-tune ourselves on that dataset.
TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)

# CLIP's statistics, used by the similarity embedding model.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class PreprocessConfig:
    """How to turn an image into a tensor for one specific model.

    This lives alongside the model in the registry, so a model and its
    preprocessing can never drift apart.

    Attributes:
        size: Target ``(height, width)`` the model expects.
        mean / std: Per-channel normalisation constants.
        resize_mode: ``"center_crop"`` for classifiers (resize shortest side,
            then crop the middle), ``"letterbox"`` for detectors (preserve
            aspect ratio, pad the remainder), ``"stretch"`` to squash to size.
        crop_pct: For ``center_crop``, resize to ``size / crop_pct`` first.
            0.875 is the torchvision convention (256 -> crop 224).
        pad_value: Grey value used to fill letterbox padding.
        to_float: Scale pixels from 0-255 into 0-1 before normalising.
        channels_first: Emit ``CHW`` (PyTorch/ONNX) rather than ``HWC``.
    """

    size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    resize_mode: str = "center_crop"
    crop_pct: float = 0.875
    pad_value: int = 114  # the value YOLO uses, kept for consistency
    to_float: bool = True
    channels_first: bool = True
    normalize: bool = True


@dataclass
class PreprocessResult:
    """A preprocessed image plus everything needed to undo the geometry.

    Detection models need ``scale`` and ``pad`` to map predicted boxes back
    onto the coordinates of the image the user actually uploaded.
    """

    array: np.ndarray
    original_size: tuple[int, int]  # (width, height)
    processed_size: tuple[int, int]  # (width, height)
    scale: float = 1.0
    pad: tuple[float, float] = (0.0, 0.0)  # (pad_x, pad_y) applied to the left/top
    warnings: list[str] = field(default_factory=list)


def decode_image(data: bytes) -> Image.Image:
    """Decode bytes into an RGB :class:`PIL.Image.Image`.

    Two corrections are applied that are easy to forget and cause subtle
    accuracy loss:

    * **EXIF orientation.** Phone photos store "rotate 90° on display" in
      metadata rather than rotating the pixels. Without this, a portrait photo
      reaches the model sideways.
    * **Colour mode.** Greyscale, palette and RGBA images are all converted to
      3-channel RGB, because models expect exactly 3 input channels. Alpha is
      composited onto white rather than dropped, which avoids black fringes.

    Raises:
        InvalidImageError: The bytes cannot be decoded.
    """
    img: Image.Image
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode now, so errors surface here
    except Exception as exc:
        raise InvalidImageError(
            "The image could not be decoded.",
            internal_message=f"{type(exc).__name__}: {exc}",
        ) from exc

    try:
        img = ImageOps.exif_transpose(img) or img
    except Exception:  # corrupt EXIF should never fail the whole request
        logger.debug("exif_transpose_failed", extra={"mode": img.mode})

    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def resize_center_crop(
    img: Image.Image, size: tuple[int, int], crop_pct: float = 0.875
) -> Image.Image:
    """Resize the shortest side, then crop the centre — the classifier recipe.

    Why not just stretch to 224x224? Because the model was *evaluated* this
    way during training. Matching the evaluation transform exactly is worth
    1-2 points of top-1 accuracy for free.
    """
    target_h, target_w = size
    resize_h = int(round(target_h / crop_pct))
    resize_w = int(round(target_w / crop_pct))

    w, h = img.size
    scale = max(resize_w / w, resize_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def resize_letterbox(
    img: Image.Image, size: tuple[int, int], pad_value: int = 114
) -> tuple[Image.Image, float, tuple[float, float]]:
    """Resize preserving aspect ratio, padding the leftover space.

    Detectors need this. If you squash a 1920x1080 photo into a 640x640
    square, every object is horizontally compressed and the predicted boxes
    are wrong in a way that is hard to notice and impossible to undo.

    Returns:
        ``(image, scale, (pad_x, pad_y))`` — the scale and padding are needed
        to map predicted boxes back to original image coordinates.
    """
    target_h, target_w = size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    pad_x = (target_w - new_w) / 2
    pad_y = (target_h - new_h) / 2

    canvas = Image.new("RGB", (target_w, target_h), (pad_value,) * 3)
    canvas.paste(img, (int(round(pad_x)), int(round(pad_y))))
    return canvas, scale, (pad_x, pad_y)


def to_array(img: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """Convert a PIL image into the normalised float32 array a model expects."""
    arr: np.ndarray = np.asarray(img, dtype=np.float32 if cfg.to_float else np.uint8)

    if cfg.to_float:
        arr = arr / 255.0
        if cfg.normalize:
            arr = (arr - np.array(cfg.mean, dtype=np.float32)) / np.array(cfg.std, dtype=np.float32)

    if cfg.channels_first:
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

    return np.ascontiguousarray(arr, dtype=np.float32 if cfg.to_float else np.uint8)


def preprocess(data: bytes, cfg: PreprocessConfig) -> PreprocessResult:
    """Full pipeline: raw bytes to a batched, model-ready array.

    Args:
        data: Validated image bytes (run :func:`~api.utils.validators.
            validate_image_bytes` first).
        cfg: The target model's preprocessing configuration.

    Returns:
        :class:`PreprocessResult` whose ``array`` has a leading batch
        dimension of 1, e.g. shape ``(1, 3, 224, 224)``.
    """
    img = decode_image(data)
    original_size = img.size
    warnings: list[str] = []

    scale, pad = 1.0, (0.0, 0.0)
    if cfg.resize_mode == "letterbox":
        img, scale, pad = resize_letterbox(img, cfg.size, cfg.pad_value)
    elif cfg.resize_mode == "center_crop":
        img = resize_center_crop(img, cfg.size, cfg.crop_pct)
    elif cfg.resize_mode == "stretch":
        img = img.resize((cfg.size[1], cfg.size[0]), Image.Resampling.BILINEAR)
    else:
        raise ValueError(f"unknown resize_mode: {cfg.resize_mode!r}")

    if original_size != img.size:
        warnings.append(
            f"image resized from {original_size[0]}x{original_size[1]} "
            f"to {img.size[0]}x{img.size[1]} using {cfg.resize_mode}"
        )

    arr = to_array(img, cfg)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension

    return PreprocessResult(
        array=arr,
        original_size=original_size,
        processed_size=img.size,
        scale=scale,
        pad=pad,
        warnings=warnings,
    )


def preprocess_batch(
    images: list[bytes], cfg: PreprocessConfig
) -> tuple[np.ndarray, list[PreprocessResult]]:
    """Preprocess several images and stack them into one batched array.

    Batching matters for throughput: one forward pass over 16 images is
    substantially cheaper than 16 separate passes, because the fixed per-call
    overhead is paid once.
    """
    results = [preprocess(data, cfg) for data in images]
    stacked = np.concatenate([r.array for r in results], axis=0)
    return stacked, results


def scale_boxes_to_original(
    boxes: np.ndarray,
    result: PreprocessResult,
) -> np.ndarray:
    """Map detector boxes from letterboxed space back to the original image.

    The transform is the exact inverse of :func:`resize_letterbox`: subtract
    the padding that was added, then divide by the resize scale. Boxes are
    finally clipped to the image bounds, because a detector can legitimately
    predict a box extending slightly past the edge.

    Args:
        boxes: Array of shape ``(N, 4)`` in ``x1, y1, x2, y2`` order, in the
            coordinate space of the preprocessed (letterboxed) image.
        result: The :class:`PreprocessResult` from preprocessing that image.

    Returns:
        Boxes of the same shape, in original-image pixel coordinates.
    """
    if boxes.size == 0:
        return boxes

    pad_x, pad_y = result.pad
    scale = result.scale or 1.0

    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale

    orig_w, orig_h = result.original_size
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out


def image_hash(data: bytes) -> str:
    """Stable content hash of an image, used as a cache key.

    Hashing the *bytes* means two identical uploads hit the cache even when
    they arrive with different filenames or from different users.
    """
    import hashlib

    return hashlib.sha256(data).hexdigest()


# Ready-made configurations for the three shipped model families.
CLASSIFICATION_PREPROCESS = PreprocessConfig(
    size=(224, 224),
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    resize_mode="center_crop",
    crop_pct=0.875,
)

# Must mirror EvalTransform in models/training/augmentation.py EXACTLY.
#
# `stretch` (a direct resize), not `center_crop`: the model is trained and
# validated on a straight resize to 64x64, so serving must do the same.
# A center crop at crop_pct=0.875 would upscale to 73px and crop back to 64,
# discarding the image border and softening the rest - the model would score
# measurably worse in production than in validation, with nothing to indicate
# why. `tests/unit/test_preprocessing_parity.py` asserts the two stay aligned.
#
# (The ImageNet config above legitimately DOES use resize-256-crop-224,
# because that is the transform its pretrained weights were evaluated with.
# The right rule is not "always crop" or "never crop" - it is "match how the
# model was evaluated".)
# 224, not Tiny-ImageNet's native 64.
#
# The images are 64x64, so this upsamples them, and that is deliberate. At
# 64px the ResNet stem has to be replaced (stride-2 7x7 -> stride-1 3x3,
# maxpool removed) or the first residual block sees a 16x16 map - and
# replacing the stem discards pretrained weights. Feeding 224px keeps the
# original ImageNet stem and runs the backbone at the resolution its features
# were learned at, so the whole pretrained network transfers unchanged.
#
# Upsampling adds no information: the source is still 64x64, which is why the
# gain over 128px is a little over one point rather than the several a
# resolution change usually buys. The ceiling here is the dataset, not the
# input size. See docs/ASSUMPTIONS.md.
#
# This MUST equal the --image-size the checkpoint was trained with.
# tests/unit/test_preprocessing_parity.py derives its expectations from this
# constant rather than hard-coding a number, so the two cannot drift apart.
TINY_IMAGENET_PREPROCESS = PreprocessConfig(
    size=(224, 224),
    mean=TINY_IMAGENET_MEAN,
    std=TINY_IMAGENET_STD,
    resize_mode="stretch",
)

DETECTION_PREPROCESS = PreprocessConfig(
    size=(640, 640),
    resize_mode="letterbox",
    normalize=False,  # YOLO normalises internally; it wants plain 0-1 input
    pad_value=114,
)

SIMILARITY_PREPROCESS = PreprocessConfig(
    size=(224, 224),
    mean=CLIP_MEAN,
    std=CLIP_STD,
    resize_mode="center_crop",
    crop_pct=1.0,
)


__all__ = [
    "CLASSIFICATION_PREPROCESS",
    "CLIP_MEAN",
    "CLIP_STD",
    "DETECTION_PREPROCESS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SIMILARITY_PREPROCESS",
    "TINY_IMAGENET_MEAN",
    "TINY_IMAGENET_PREPROCESS",
    "TINY_IMAGENET_STD",
    "PreprocessConfig",
    "PreprocessResult",
    "decode_image",
    "image_hash",
    "preprocess",
    "preprocess_batch",
    "resize_center_crop",
    "resize_letterbox",
    "scale_boxes_to_original",
    "to_array",
]
