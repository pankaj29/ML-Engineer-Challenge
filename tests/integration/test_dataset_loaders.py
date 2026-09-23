"""Regression tests for Tiny-ImageNet label correctness.

These exist because of a real bug in the provided starter script: it loaded
the validation split with ``ImageFolder``, which cannot read Tiny-ImageNet's
layout and silently assigned **label 0 to all 10,000 validation images**.

It did not crash. Validation accuracy simply read a meaningless ~0.5%, which
looks exactly like a broken training loop.

The defining property of this bug is that it is *silent*, so the tests below
assert the things that would have caught it:

* the validation split spans many classes, not one;
* every class has the expected number of images;
* the training and validation splits share one class-index mapping.

Both loaders are covered — the fixed starter script and the fuller one used by
the training pipeline — because both are shipped and either could regress.

They skip when the dataset is absent, so a fresh clone still has a green suite:

    python scripts/download_datasets.py --dataset tiny_imagenet --data-dir data
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = [pytest.mark.integration, pytest.mark.slow]

EXPECTED_CLASSES = 200
EXPECTED_TRAIN_IMAGES = 100_000
EXPECTED_VAL_IMAGES = 10_000
EXPECTED_VAL_PER_CLASS = 50


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    """Locate the extracted dataset, or skip."""
    from models.training.dataset import find_dataset_root

    try:
        return find_dataset_root(REPO_ROOT / "data")
    except FileNotFoundError:
        pytest.skip(
            "Tiny-ImageNet is not downloaded; run "
            "`python scripts/download_datasets.py --dataset tiny_imagenet --data-dir data`"
        )


class TestTrainingPipelineLoader:
    """models/training/dataset.py — what the training pipeline uses."""

    def test_train_split_is_complete(self, dataset_root: Path) -> None:
        from models.training.dataset import TinyImageNetTrain

        train = TinyImageNetTrain(dataset_root)
        assert len(train.classes) == EXPECTED_CLASSES
        assert len(train) == EXPECTED_TRAIN_IMAGES

    def test_val_labels_are_not_degenerate(self, dataset_root: Path) -> None:
        """THE regression test: the validation split must span many classes.

        With the original ImageFolder approach this returns 1.
        """
        from models.training.dataset import TinyImageNetTrain, TinyImageNetVal

        train = TinyImageNetTrain(dataset_root)
        val = TinyImageNetVal(dataset_root, train.class_to_idx)

        distinct = len({label for _, label in val.samples})
        assert distinct == EXPECTED_CLASSES, (
            f"validation split has {distinct} distinct label(s), expected "
            f"{EXPECTED_CLASSES}. A value of 1 means the loader is reading the "
            "directory name instead of val_annotations.txt."
        )

    def test_val_split_is_balanced(self, dataset_root: Path) -> None:
        from models.training.dataset import TinyImageNetTrain, TinyImageNetVal

        train = TinyImageNetTrain(dataset_root)
        val = TinyImageNetVal(dataset_root, train.class_to_idx)

        assert len(val) == EXPECTED_VAL_IMAGES
        counts = collections.Counter(label for _, label in val.samples)
        assert set(counts.values()) == {EXPECTED_VAL_PER_CLASS}

    def test_splits_share_one_class_mapping(self, dataset_root: Path) -> None:
        """Two independently-built mappings could order classes differently,
        which would silently mislabel every validation image."""
        from models.training.dataset import TinyImageNetTrain, TinyImageNetVal

        train = TinyImageNetTrain(dataset_root)
        val = TinyImageNetVal(dataset_root, train.class_to_idx)
        assert val.class_to_idx is train.class_to_idx

    def test_class_subset_filters_both_splits(self, dataset_root: Path) -> None:
        """A class subset must drop validation images of excluded classes,
        not relabel them."""
        from models.training.dataset import TinyImageNetTrain, TinyImageNetVal

        train = TinyImageNetTrain(dataset_root, class_subset=10)
        val = TinyImageNetVal(dataset_root, train.class_to_idx)

        assert len(train.classes) == 10
        assert len(val) == 10 * EXPECTED_VAL_PER_CLASS
        assert max(label for _, label in val.samples) == 9

    def test_class_names_are_readable(self, dataset_root: Path) -> None:
        """Serving returns 'goldfish', not 'n01443537'."""
        from models.training.dataset import TinyImageNetTrain, load_class_names

        train = TinyImageNetTrain(dataset_root, class_subset=5)
        names = load_class_names(dataset_root)
        readable = [names.get(w, w) for w in train.classes]

        assert readable[0] == "goldfish"
        assert not any(name.startswith("n0") for name in readable)


class TestProvidedStarterLoader:
    """scripts/tiny_imagenet_dataloader.py — the fixed starter script.

    Shipped with the project, so it gets the same guarantees.
    """

    @pytest.fixture
    def loaders(self, dataset_root: Path):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from tiny_imagenet_dataloader import get_tiny_imagenet_dataloaders

        return get_tiny_imagenet_dataloaders(str(dataset_root), batch_size=32, num_workers=0)

    def test_split_sizes(self, loaders) -> None:
        train_loader, val_loader = loaders
        assert len(train_loader.dataset) == EXPECTED_TRAIN_IMAGES
        assert len(val_loader.dataset) == EXPECTED_VAL_IMAGES

    def test_val_labels_are_not_degenerate(self, loaders) -> None:
        """The original bug, asserted directly against the starter script."""
        _, val_loader = loaders
        distinct = len({label for _, label in val_loader.dataset.samples})
        assert distinct == EXPECTED_CLASSES, (
            f"validation split has {distinct} distinct label(s). The starter "
            "script has regressed to using ImageFolder on val/."
        )

    def test_val_split_is_balanced(self, loaders) -> None:
        _, val_loader = loaders
        counts = collections.Counter(label for _, label in val_loader.dataset.samples)
        assert set(counts.values()) == {EXPECTED_VAL_PER_CLASS}

    def test_produces_usable_tensors(self, loaders) -> None:
        _, val_loader = loaders
        image, label = val_loader.dataset[0]
        assert tuple(image.shape) == (3, 64, 64)
        assert 0 <= label < EXPECTED_CLASSES

    def test_missing_annotations_file_raises_clearly(self, tmp_path) -> None:
        """A missing annotations file must fail loudly, not fall back to
        something that produces wrong labels."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from tiny_imagenet_dataloader import TinyImageNetVal

        (tmp_path / "val" / "images").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match=r"val_annotations\.txt"):
            TinyImageNetVal(str(tmp_path), class_to_idx={"n01443537": 0})


class TestFullDatasetGuard:
    """Training must refuse to run on anything less than the whole dataset.

    The subsetting options (`--classes`, `--limit-batches`) were removed from
    the training entrypoint because a run that stops after 3 of 781 batches
    still prints an epoch summary and writes a checkpoint — there is nothing
    in the output to distinguish a smoke test from a real trained model.

    `verify_full_dataset()` replaces them with a hard gate. These tests prove
    it fires.
    """

    @pytest.fixture
    def aug(self):
        from models.training.augmentation import AugmentationConfig

        return AugmentationConfig(image_size=64)

    def _loaders(self, aug, class_subset=None):
        from models.training.augmentation import EvalTransform, TrainTransform
        from models.training.dataset import build_dataloaders

        train_loader, val_loader, _, _ = build_dataloaders(
            REPO_ROOT / "data",
            batch_size=128,
            num_workers=0,
            train_transform=TrainTransform(aug),
            eval_transform=EvalTransform(aug),
            class_subset=class_subset,
        )
        return train_loader, val_loader

    def test_accepts_the_complete_dataset(self, dataset_root: Path, aug) -> None:
        from models.training.train_classifier import verify_full_dataset

        train_loader, val_loader = self._loaders(aug)
        verified = verify_full_dataset(REPO_ROOT / "data", train_loader, val_loader)

        assert verified["classes"] == EXPECTED_CLASSES
        assert verified["train_images"] == EXPECTED_TRAIN_IMAGES
        assert verified["val_images"] == EXPECTED_VAL_IMAGES
        assert verified["val_labels"] == EXPECTED_CLASSES

    def test_rejects_a_class_subset(self, dataset_root: Path, aug) -> None:
        """The exact scenario the guard exists for."""
        from models.training.train_classifier import verify_full_dataset

        train_loader, val_loader = self._loaders(aug, class_subset=10)
        with pytest.raises(RuntimeError, match="incomplete dataset"):
            verify_full_dataset(REPO_ROOT / "data", train_loader, val_loader)

    def test_error_names_what_is_missing(self, dataset_root: Path, aug) -> None:
        """An error that says 'something is wrong' wastes the reader's time."""
        from models.training.train_classifier import verify_full_dataset

        train_loader, val_loader = self._loaders(aug, class_subset=10)
        with pytest.raises(RuntimeError) as exc:
            verify_full_dataset(REPO_ROOT / "data", train_loader, val_loader)

        message = str(exc.value)
        assert "10 of 200 classes" in message
        assert "95,000 missing" in message
        assert "download_datasets.py" in message

    def test_full_epoch_covers_every_image(self, dataset_root: Path, aug) -> None:
        """Batch counts must account for the whole dataset.

        `drop_last=True` on the training loader can discard a final partial
        batch, so allow one batch of slack there but none on validation.
        """
        train_loader, val_loader = self._loaders(aug)

        assert len(train_loader) * 128 > EXPECTED_TRAIN_IMAGES - 128
        assert len(val_loader) * 128 >= EXPECTED_VAL_IMAGES


class TestNoSubsettingInTrainingCLI:
    """The removed flags must stay removed."""

    def test_train_config_has_no_subsetting_fields(self) -> None:
        from models.training.train_classifier import TrainConfig

        fields = set(TrainConfig.__dataclass_fields__)
        assert "class_subset" not in fields
        assert "limit_batches" not in fields

    def test_cli_rejects_the_removed_flags(self) -> None:
        """Passing --classes must fail, not be silently ignored."""
        import subprocess

        for flag in ("--classes", "--limit-batches"):
            result = subprocess.run(
                [sys.executable, "-m", "models.training.train_classifier", flag, "10"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                timeout=120,
            )
            assert result.returncode != 0
            assert (
                "unrecognized arguments" in result.stderr
            ), f"{flag} was accepted; subsetting has crept back in"
