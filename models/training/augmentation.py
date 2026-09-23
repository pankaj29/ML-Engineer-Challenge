"""Custom data augmentation pipeline.

Plain English:
    A model that sees the same 100,000 images every epoch memorises them.
    Augmentation makes each epoch look slightly different — flipped, shifted,
    re-coloured, partly hidden — so the model is forced to learn what a
    goldfish *is* rather than which pixels the goldfish photo had.

    The result is better accuracy on images it has never seen, which is the
    only accuracy that matters.

This module implements the pipeline by hand rather than importing a library,
because the challenge asks for a *custom* pipeline. Each transform below says
what it does and, more importantly, **why it helps** and **when it hurts**.

The heavy hitters are the last two, and they are the ones people usually skip:

* **RandAugment** — rather than hand-tuning a dozen knobs, pick N random
  operations from a fixed menu at a single shared magnitude. It removes the
  guesswork and reliably beats hand-tuned pipelines.
* **MixUp / CutMix** — blend two *different* images and their labels together.
  This sounds absurd and works remarkably well: it stops the network being
  over-confident, because it is routinely asked to predict "70% cat, 30% dog".

Ordering matters. Geometric transforms run first (on the PIL image), then
colour transforms, then the tensor conversion, then erasing. Applying a colour
jitter after normalisation, for example, would shift the carefully matched
mean and standard deviation and quietly cost accuracy.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import default_collate


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------
def _blend(img: Image.Image, enhancer: type, magnitude: float) -> Image.Image:
    """Apply a PIL enhancer where 1.0 means 'no change'."""
    return enhancer(img).enhance(magnitude)


def autocontrast(img: Image.Image, _: float) -> Image.Image:
    """Stretch the histogram to use the full range. Helps flat, hazy photos."""
    return ImageOps.autocontrast(img)


def equalize(img: Image.Image, _: float) -> Image.Image:
    """Equalise the histogram. Strong effect; good for varied exposure."""
    return ImageOps.equalize(img)


def invert(img: Image.Image, _: float) -> Image.Image:
    """Invert colours. Aggressive, and deliberately rare in the menu below."""
    return ImageOps.invert(img)


def rotate(img: Image.Image, magnitude: float) -> Image.Image:
    """Rotate by up to +/-30 degrees. Teaches tolerance to camera tilt."""
    degrees = magnitude * 30.0 * random.choice([-1, 1])
    return img.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))


def posterize(img: Image.Image, magnitude: float) -> Image.Image:
    """Reduce colour depth, forcing reliance on shape rather than exact hue."""
    bits = max(1, int(8 - magnitude * 4))
    return ImageOps.posterize(img, bits)


def solarize(img: Image.Image, magnitude: float) -> Image.Image:
    """Invert pixels above a threshold. An odd but effective colour shift."""
    threshold = int(256 - magnitude * 200)
    return ImageOps.solarize(img, threshold)


def adjust_color(img: Image.Image, magnitude: float) -> Image.Image:
    """Change saturation, from greyscale through to oversaturated."""
    return _blend(img, ImageEnhance.Color, 1.0 + magnitude * random.choice([-0.9, 1.0]))


def adjust_contrast(img: Image.Image, magnitude: float) -> Image.Image:
    """Change contrast. Simulates different cameras and lighting."""
    return _blend(img, ImageEnhance.Contrast, 1.0 + magnitude * random.choice([-0.9, 1.0]))


def adjust_brightness(img: Image.Image, magnitude: float) -> Image.Image:
    """Change brightness. Simulates under- and over-exposure."""
    return _blend(img, ImageEnhance.Brightness, 1.0 + magnitude * random.choice([-0.9, 1.0]))


def adjust_sharpness(img: Image.Image, magnitude: float) -> Image.Image:
    """Blur or sharpen. Simulates focus quality and image compression."""
    return _blend(img, ImageEnhance.Sharpness, 1.0 + magnitude * random.choice([-0.9, 1.0]))


def shear_x(img: Image.Image, magnitude: float) -> Image.Image:
    """Slant horizontally. A mild viewpoint change."""
    factor = magnitude * 0.3 * random.choice([-1, 1])
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, factor, 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(128, 128, 128),
    )


def shear_y(img: Image.Image, magnitude: float) -> Image.Image:
    """Slant vertically."""
    factor = magnitude * 0.3 * random.choice([-1, 1])
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, 0, factor, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(128, 128, 128),
    )


def translate_x(img: Image.Image, magnitude: float) -> Image.Image:
    """Shift horizontally. Teaches that position does not change identity."""
    pixels = magnitude * img.size[0] * 0.3 * random.choice([-1, 1])
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, pixels, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(128, 128, 128),
    )


def translate_y(img: Image.Image, magnitude: float) -> Image.Image:
    """Shift vertically."""
    pixels = magnitude * img.size[1] * 0.3 * random.choice([-1, 1])
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, 0, 0, 1, pixels),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(128, 128, 128),
    )


# The RandAugment menu. `invert` is deliberately absent: on natural photos it
# is destructive far more often than it is useful.
RANDAUGMENT_OPS: list[Callable[[Image.Image, float], Image.Image]] = [
    autocontrast,
    equalize,
    rotate,
    posterize,
    solarize,
    adjust_color,
    adjust_contrast,
    adjust_brightness,
    adjust_sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]


class RandAugment:
    """Apply ``num_ops`` randomly chosen operations at a fixed magnitude.

    Args:
        num_ops: How many operations to apply per image. 2 is the standard
            choice and a good default.
        magnitude: Strength from 0 (no-op) to 1 (maximum). Around 0.3-0.5
            suits small datasets; push higher only with a big dataset and a
            long schedule, or the model spends its capacity fighting noise.
    """

    def __init__(self, num_ops: int = 2, magnitude: float = 0.4) -> None:
        self.num_ops = num_ops
        self.magnitude = magnitude

    def __call__(self, img: Image.Image) -> Image.Image:
        for op in random.sample(RANDAUGMENT_OPS, k=min(self.num_ops, len(RANDAUGMENT_OPS))):
            # Jitter the magnitude per operation so the pipeline does not
            # produce the same intensity every time.
            img = op(img, random.uniform(0.1, self.magnitude))
        return img

    def __repr__(self) -> str:
        return f"RandAugment(num_ops={self.num_ops}, magnitude={self.magnitude})"


class RandomResizedCropCustom:
    """Crop a random region, then resize it to the target size.

    This is the single most valuable augmentation for classification. By
    showing the model a different sub-region each epoch it learns that a
    goldfish is a goldfish whether it fills the frame or sits in one corner.

    Args:
        size: Output ``(height, width)``.
        scale: Fraction of the original *area* to keep. The default lower
            bound of 0.35 is deliberately gentler than torchvision's 0.08:
            Tiny-ImageNet images are only 64x64, and cropping 8% of that
            leaves 18x18 pixels, which frequently contains no subject at all
            while still carrying the original label. That is label noise, not
            augmentation.
        ratio: Range of allowed aspect ratios for the crop.
    """

    def __init__(
        self,
        size: tuple[int, int] = (64, 64),
        scale: tuple[float, float] = (0.35, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        area = width * height

        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect = math.exp(random.uniform(*log_ratio))

            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))

            if crop_w <= width and crop_h <= height:
                left = random.randint(0, width - crop_w)
                top = random.randint(0, height - crop_h)
                img = img.crop((left, top, left + crop_w, top + crop_h))
                return img.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)

        # Ten failed attempts: fall back to a centre crop rather than looping
        # forever on an extreme aspect ratio.
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        img = img.crop((left, top, left + side, top + side))
        return img.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)


class RandomErasing:
    """Blank out a random rectangle of the image.

    Simulates occlusion — an object partly behind something else. Without it,
    a model can become dependent on one distinctive patch always being
    visible, and falls apart when that patch is hidden.

    Applied to the tensor *after* normalisation, filling with random noise
    (which, in normalised space, is near the dataset mean).
    """

    def __init__(
        self,
        probability: float = 0.25,
        scale: tuple[float, float] = (0.02, 0.2),
        ratio: tuple[float, float] = (0.3, 3.3),
    ) -> None:
        self.probability = probability
        self.scale = scale
        self.ratio = ratio

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return tensor

        _, height, width = tensor.shape
        area = height * width

        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect = random.uniform(*self.ratio)
            erase_h = int(round(math.sqrt(target_area * aspect)))
            erase_w = int(round(math.sqrt(target_area / aspect)))

            if erase_h < height and erase_w < width:
                top = random.randint(0, height - erase_h)
                left = random.randint(0, width - erase_w)
                tensor[:, top : top + erase_h, left : left + erase_w] = torch.randn(
                    tensor.shape[0], erase_h, erase_w
                )
                return tensor
        return tensor


# ---------------------------------------------------------------------------
# Composed pipelines
# ---------------------------------------------------------------------------
@dataclass
class AugmentationConfig:
    """Knobs for the training augmentation pipeline."""

    image_size: int = 64
    mean: tuple[float, float, float] = (0.4802, 0.4481, 0.3975)
    std: tuple[float, float, float] = (0.2302, 0.2265, 0.2262)

    random_resized_crop: bool = True
    crop_scale: tuple[float, float] = (0.35, 1.0)
    horizontal_flip_prob: float = 0.5
    # Vertical flip is off by default: most real-world photos have a
    # consistent "up", and teaching the model that upside-down is normal
    # costs accuracy on everything except aerial or microscopy imagery.
    vertical_flip_prob: float = 0.0
    randaugment: bool = True
    randaugment_ops: int = 2
    randaugment_magnitude: float = 0.4
    random_erasing_prob: float = 0.25

    # Batch-level augmentation, applied in the collate function.
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5  # chance that a batch gets MixUp or CutMix at all

    label_smoothing: float = 0.1

    def describe(self) -> dict[str, Any]:
        """Serialisable summary, recorded with training runs for reproducibility."""
        return {
            "image_size": self.image_size,
            "random_resized_crop": self.random_resized_crop,
            "crop_scale": list(self.crop_scale),
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "randaugment": (
                f"{self.randaugment_ops} ops @ magnitude {self.randaugment_magnitude}"
                if self.randaugment
                else "off"
            ),
            "random_erasing_prob": self.random_erasing_prob,
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
            "mix_prob": self.mix_prob,
            "label_smoothing": self.label_smoothing,
        }


class TrainTransform:
    """The full training-time augmentation pipeline.

    Order: geometric -> colour -> tensor -> normalise -> erase. See the module
    docstring for why this order is not interchangeable.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        size = (self.config.image_size, self.config.image_size)

        self.crop = (
            RandomResizedCropCustom(size, scale=self.config.crop_scale)
            if self.config.random_resized_crop
            else None
        )
        self.randaugment = (
            RandAugment(self.config.randaugment_ops, self.config.randaugment_magnitude)
            if self.config.randaugment
            else None
        )
        self.erasing = (
            RandomErasing(self.config.random_erasing_prob)
            if self.config.random_erasing_prob > 0
            else None
        )
        self._mean = torch.tensor(self.config.mean).view(3, 1, 1)
        self._std = torch.tensor(self.config.std).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 1. Geometry
        if self.crop is not None:
            img = self.crop(img)
        elif img.size != (self.config.image_size, self.config.image_size):
            img = img.resize(
                (self.config.image_size, self.config.image_size), Image.Resampling.BILINEAR
            )

        if random.random() < self.config.horizontal_flip_prob:
            img = ImageOps.mirror(img)
        if random.random() < self.config.vertical_flip_prob:
            img = ImageOps.flip(img)

        # 2. Colour / photometric
        if self.randaugment is not None:
            img = self.randaugment(img)

        # 3. To normalised tensor
        tensor = torch.from_numpy(np.asarray(img, dtype=np.float32).copy() / 255.0)
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - self._mean) / self._std

        # 4. Occlusion, after normalisation
        if self.erasing is not None:
            tensor = self.erasing(tensor)

        return tensor


class EvalTransform:
    """Validation-time transform: deterministic, no randomness at all.

    This is important and often got wrong. Augmenting the validation set makes
    the score noisy and not comparable between epochs — you can no longer tell
    whether the model improved or simply got an easier random crop.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        self._mean = torch.tensor(self.config.mean).view(3, 1, 1)
        self._std = torch.tensor(self.config.std).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        size = self.config.image_size
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(img, dtype=np.float32).copy() / 255.0)
        tensor = tensor.permute(2, 0, 1)
        return (tensor - self._mean) / self._std


# ---------------------------------------------------------------------------
# Batch-level augmentation: MixUp and CutMix
# ---------------------------------------------------------------------------
def mixup_batch(
    images: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Blend each image with another image from the same batch.

    The blend weight ``lam`` is drawn from a Beta distribution, which mostly
    produces values near 0 or 1 (a slight blend) and occasionally near 0.5 (a
    true half-and-half). The *labels* are blended by the same weight, which is
    what teaches the model to express uncertainty.

    Returns:
        ``(mixed_images, targets_a, targets_b, lam)``. The loss is then
        ``lam * loss(pred, a) + (1 - lam) * loss(pred, b)``.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def cutmix_batch(
    images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Paste a rectangle from another image over this one.

    Where MixUp produces a ghostly double-exposure, CutMix produces a collage:
    a real patch of image B stamped onto image A. The label weight is the
    *actual pasted area*, recomputed after clipping, not the value drawn from
    the Beta distribution — a box clipped at the image edge covers less area
    than intended, and using the intended value would mislabel the sample.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=images.device)

    _, _, height, width = images.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)

    cx, cy = random.randint(0, width), random.randint(0, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)

    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return images, targets, targets[index], lam


class MixCollate:
    """Collate function that applies MixUp or CutMix to each batch.

    Placing this in the collate step rather than the training loop means the
    augmentation happens inside the DataLoader's worker processes, in parallel
    with the GPU's work on the previous batch, instead of stalling training.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(self, batch: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        images, targets = default_collate(batch)

        use_mix = (
            random.random() < self.config.mix_prob
            and images.size(0) > 1
            and (self.config.mixup_alpha > 0 or self.config.cutmix_alpha > 0)
        )
        if not use_mix:
            # lam = 1.0 means "no mixing"; the training loop handles this
            # uniformly, so there is no special case in the loss.
            return images, targets, targets, 1.0

        if self.config.cutmix_alpha > 0 and (self.config.mixup_alpha <= 0 or random.random() < 0.5):
            return cutmix_batch(images, targets, self.config.cutmix_alpha)
        return mixup_batch(images, targets, self.config.mixup_alpha)


def mix_criterion(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    predictions: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Loss for a mixed batch: weighted sum of the two label losses."""
    if lam >= 1.0:
        return criterion(predictions, targets_a)
    return lam * criterion(predictions, targets_a) + (1.0 - lam) * criterion(predictions, targets_b)


__all__ = [
    "RANDAUGMENT_OPS",
    "AugmentationConfig",
    "EvalTransform",
    "MixCollate",
    "RandAugment",
    "RandomErasing",
    "RandomResizedCropCustom",
    "TrainTransform",
    "cutmix_batch",
    "mix_criterion",
    "mixup_batch",
]
