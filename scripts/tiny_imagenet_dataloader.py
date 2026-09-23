"""Tiny-ImageNet dataloaders.

This is the starter script provided with the challenge, with one **bug fix**.

THE BUG (and why it matters)
----------------------------
The original loaded the validation split with ``ImageFolder``::

    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), ...)

Tiny-ImageNet's two splits are stored in *different* layouts:

    train/n01443537/images/n01443537_0.JPEG    <- one folder per class
    val/images/val_0.JPEG                      <- ONE flat folder
    val/val_annotations.txt                    <- the labels live in here

``ImageFolder`` infers labels from directory names. Pointed at ``val/`` it
sees a single sub-directory called ``images`` and therefore assigns
**label 0 to all 10,000 validation images**.

It does not crash. Validation accuracy simply reads a meaningless ~0.5%, and
the natural reaction is to go hunting for a bug in the training loop that is
not there. Verified against the real dataset before fixing:

    ImageFolder on val/ -> 10000 images, 1 distinct label: {0: 10000}

THE FIX
-------
:class:`TinyImageNetVal` below reads ``val_annotations.txt`` and maps each
image to its real class, reusing the **same class-to-index mapping** as the
training split so the two are directly comparable. Verified after fixing:
200 distinct labels, exactly 50 images each.

The public API (``get_tiny_imagenet_dataloaders``) is unchanged, so anything
calling this script keeps working — it just gets correct labels now.

A more fully-featured version of this loader, with class subsetting and
readable class names, lives in :mod:`models.training.dataset` and is what the
training pipeline actually uses.
"""

import os
import zipfile

import requests
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def download_and_extract_tiny_imagenet(url, dataset_path="tiny-imagenet-200"):
    # Define the download path
    download_path = f"{dataset_path}.zip"

    # Download the dataset
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(download_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    # Extract the dataset
    print(f"Extracting {download_path}...")
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    # Clean up the zip file
    os.remove(download_path)
    print(f"Dataset downloaded and extracted to {dataset_path}")


class TinyImageNetVal(Dataset):
    """Tiny-ImageNet validation split, with correct labels.

    Reads ``val/val_annotations.txt`` (a tab-separated file whose first two
    columns are the filename and its WordNet id) and maps each image to the
    class index used by the training split.

    Args:
        data_dir: Dataset root, the directory containing ``train/`` and ``val/``.
        class_to_idx: The training split's ``{wnid: index}`` mapping. Passing
            the training mapping rather than rebuilding one is essential: two
            independently-built mappings could order the classes differently,
            and every validation label would then be silently wrong.
        transform: Applied to each PIL image.
    """

    def __init__(self, data_dir, class_to_idx, transform=None):
        from PIL import Image

        self._Image = Image
        self.transform = transform
        self.class_to_idx = class_to_idx

        val_dir = os.path.join(data_dir, "val")
        annotations = os.path.join(val_dir, "val_annotations.txt")
        images_dir = os.path.join(val_dir, "images")

        if not os.path.isfile(annotations):
            raise FileNotFoundError(
                f"val_annotations.txt not found at {annotations}. Without it the "
                "validation labels cannot be recovered - see this module's docstring."
            )

        self.samples = []
        with open(annotations, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                path = os.path.join(images_dir, filename)
                if os.path.isfile(path):
                    self.samples.append((path, class_to_idx[wnid]))

        if not self.samples:
            raise RuntimeError(f"no validation images found under {val_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with self._Image.open(path) as img:
            img = img.convert("RGB")
        return (self.transform(img) if self.transform else img), target


def get_tiny_imagenet_dataloaders(data_dir, batch_size=32, num_workers=2):
    # Define the transforms for training and validation
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
        ]
    )

    # Training split IS one-folder-per-class, so ImageFolder is correct here.
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)

    # Validation split is NOT. ImageFolder would label every image 0 - see the
    # module docstring. Reuse the training split's class mapping so the indices
    # line up.
    val_dataset = TinyImageNetVal(
        data_dir, class_to_idx=train_dataset.class_to_idx, transform=transform_val
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Download and extract the dataset
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dataset_path = "tiny-imagenet-200"
    if not os.path.exists(dataset_path):
        download_and_extract_tiny_imagenet(url, dataset_path)

    # Create DataLoaders
    train_loader, val_loader = get_tiny_imagenet_dataloaders(
        dataset_path, batch_size=32, num_workers=4
    )

    # Print dataset sizes
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    # Sanity check that would have caught the original bug immediately: the
    # validation split must span many classes, not one.
    distinct = len({label for _, label in val_loader.dataset.samples})
    print(f"Distinct validation labels: {distinct}")
    if distinct <= 1:
        raise SystemExit("ERROR: validation labels are degenerate - the loader is broken")
