"""Tiny-ImageNet dataset loading.

Plain English:
    Tiny-ImageNet is 200 classes, 100,000 training images and 10,000
    validation images, all 64x64 pixels. It is ImageNet's shape at a size that
    fits on a laptop.

**The trap this module exists to avoid.** The two splits are stored in
*different layouts*:

    train/n01443537/images/n01443537_0.JPEG     <- folder per class
    val/images/val_0.JPEG                       <- one flat folder
    val/val_annotations.txt                     <- ...labels live in this file

``torchvision.datasets.ImageFolder`` understands the first layout and not the
second. Point it at ``val/`` and it either crashes or — far worse — silently
treats ``images`` as a single class, so every validation image gets label 0.
Validation accuracy then reads a meaningless 0.5% and you go looking for a bug
in your training loop that is not there.

(The starter script ``scripts/tiny_imagenet_dataloader.py`` has exactly this
bug: it calls ``ImageFolder`` on the val directory.)

This module reads ``val_annotations.txt`` and maps every validation image to
the correct class, using the *same* class-index ordering as the training split
so the two are directly comparable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset


def find_dataset_root(base: Path) -> Path:
    """Locate the real dataset root, tolerating a nested extraction.

    The download script extracts into ``data/tiny-imagenet-200/`` and the zip
    itself contains a ``tiny-imagenet-200/`` folder, producing
    ``data/tiny-imagenet-200/tiny-imagenet-200/``. Rather than hard-coding
    either shape, look for the directory that actually contains ``train``.

    Raises:
        FileNotFoundError: No Tiny-ImageNet layout found under ``base``.
    """
    base = Path(base)
    candidates = [
        base,
        base / "tiny-imagenet-200",
        base / "tiny-imagenet-200" / "tiny-imagenet-200",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "wnids.txt").is_file():
            return candidate

    # Last resort: search for it, in case the layout changes again.
    for path in base.rglob("wnids.txt"):
        if (path.parent / "train").is_dir():
            return path.parent

    raise FileNotFoundError(
        f"Tiny-ImageNet not found under {base}. Download it with:\n"
        f"  python scripts/download_datasets.py --dataset tiny_imagenet"
    )


def load_class_names(root: Path) -> dict[str, str]:
    """Map WordNet ids to readable names using ``words.txt``.

    ``n01443537`` on its own is unreadable. This turns it into ``goldfish``,
    which is what the API returns to users.
    """
    words_file = root / "words.txt"
    if not words_file.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in words_file.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        wnid, names = line.split("\t", 1)
        # words.txt lists several synonyms; the first is the common name.
        mapping[wnid.strip()] = names.split(",")[0].strip()
    return mapping


@dataclass
class DatasetStats:
    """Summary of a loaded dataset, printed before training starts."""

    num_classes: int
    train_size: int
    val_size: int
    image_size: int
    root: str

    def describe(self) -> str:
        return (
            f"{self.num_classes} classes | {self.train_size:,} train | "
            f"{self.val_size:,} val | {self.image_size}x{self.image_size} px"
        )


class TinyImageNetTrain(Dataset):
    """Training split: one sub-directory per class.

    Args:
        root: Dataset root (the directory containing ``train/``).
        transform: Callable applied to each PIL image.
        class_subset: Use only the first N classes. Useful for a fast smoke
            run on CPU without changing any other code path.
    """

    def __init__(
        self,
        root: Path,
        transform: Callable[[Image.Image], Any] | None = None,
        class_subset: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.train_dir = self.root / "train"

        if not self.train_dir.is_dir():
            raise FileNotFoundError(f"training directory not found: {self.train_dir}")

        # Sorted so the class-index mapping is deterministic. If this ordering
        # ever changed between training and serving, every prediction would be
        # labelled with the wrong class name.
        wnids = sorted(d.name for d in self.train_dir.iterdir() if d.is_dir())
        if class_subset:
            wnids = wnids[:class_subset]

        self.classes: list[str] = wnids
        self.class_to_idx: dict[str, int] = {w: i for i, w in enumerate(wnids)}

        self.samples: list[tuple[Path, int]] = []
        for wnid in wnids:
            images_dir = self.train_dir / wnid / "images"
            if not images_dir.is_dir():
                images_dir = self.train_dir / wnid
            idx = self.class_to_idx[wnid]
            for path in sorted(images_dir.glob("*.JPEG")):
                self.samples.append((path, idx))

        if not self.samples:
            raise RuntimeError(f"no training images found under {self.train_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return (self.transform(img) if self.transform else img), target


class TinyImageNetVal(Dataset):
    """Validation split: flat folder plus an annotations file.

    This is the class that exists because ``ImageFolder`` cannot read this
    layout. See the module docstring.
    """

    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = class_to_idx

        val_dir = self.root / "val"
        annotations = val_dir / "val_annotations.txt"
        images_dir = val_dir / "images"

        if not annotations.is_file():
            raise FileNotFoundError(
                f"val_annotations.txt not found at {annotations}. Without it the "
                "validation labels cannot be recovered."
            )

        self.samples: list[tuple[Path, int]] = []
        skipped = 0
        for line in annotations.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            # A class subset means most validation images belong to classes we
            # are not training on; those are skipped rather than mislabelled.
            if wnid not in class_to_idx:
                skipped += 1
                continue
            path = images_dir / filename
            if path.is_file():
                self.samples.append((path, class_to_idx[wnid]))

        self.skipped = skipped
        if not self.samples:
            raise RuntimeError(f"no validation images matched the training classes under {val_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return (self.transform(img) if self.transform else img), target


def build_dataloaders(
    data_dir: Path,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    train_transform: Callable[[Image.Image], Any] | None = None,
    eval_transform: Callable[[Image.Image], Any] | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
    class_subset: int | None = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DatasetStats, list[str]]:
    """Build training and validation dataloaders.

    Args:
        data_dir: Directory containing the extracted dataset.
        batch_size: Images per batch.
        num_workers: Loader subprocesses. 0 means load in the main process,
            which is required on Windows when this is called from a script
            without an ``if __name__ == "__main__"`` guard.
        collate_fn: Usually :class:`~models.training.augmentation.MixCollate`.
        class_subset: Train on only the first N classes, for fast iteration.

    Returns:
        ``(train_loader, val_loader, stats, readable_class_names)``.
    """
    root = find_dataset_root(data_dir)

    train_ds = TinyImageNetTrain(root, transform=train_transform, class_subset=class_subset)
    val_ds = TinyImageNetVal(root, train_ds.class_to_idx, transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        # Dropping the final partial batch keeps every batch the same size,
        # which matters for BatchNorm: a last batch of 1 image produces
        # meaningless batch statistics.
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    names = load_class_names(root)
    readable = [names.get(wnid, wnid) for wnid in train_ds.classes]

    stats = DatasetStats(
        num_classes=len(train_ds.classes),
        train_size=len(train_ds),
        val_size=len(val_ds),
        image_size=64,
        root=str(root),
    )
    return train_loader, val_loader, stats, readable


def save_labels(names: list[str], path: Path) -> Path:
    """Write the ordered class-name list next to the model artifact.

    The labels must travel with the model. A checkpoint without its label list
    can still predict "class 137", but nobody can say what 137 means.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(names), encoding="utf-8")
    return path


__all__ = [
    "DatasetStats",
    "TinyImageNetTrain",
    "TinyImageNetVal",
    "build_dataloaders",
    "find_dataset_root",
    "load_class_names",
    "save_labels",
]
