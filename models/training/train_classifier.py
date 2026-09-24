"""Fine-tune an image classifier on Tiny-ImageNet.

Plain English:
    Take a model that already knows how to see (pretrained on ImageNet) and
    teach it the 200 Tiny-ImageNet classes. Starting from pretrained weights
    rather than from scratch is the difference between a few epochs and a few
    hundred.

The challenge requires three specific techniques. Each is explained where it
is implemented, but in short:

1. **Mixed precision** — do the maths in 16-bit instead of 32-bit where it is
   safe to. Roughly 2x faster on a modern GPU and uses about half the memory,
   with no measurable accuracy cost. The catch is that 16-bit numbers underflow
   to zero easily, which would silently destroy small gradients; a *gradient
   scaler* multiplies the loss up before the backward pass and divides the
   gradients back down afterwards to prevent that.

2. **Gradient clipping** — cap how large a single weight update can be. One
   unlucky batch can produce an enormous gradient that throws the weights into
   a region they never recover from; the loss jumps to NaN and the run is
   dead. Clipping bounds the damage.

3. **Learning rate scheduling** — start small (warmup), rise, then decay
   smoothly to near zero (cosine). Warmup matters most with pretrained
   weights: a large learning rate in the first few hundred steps will wash out
   exactly the features you are trying to keep.

**Training always uses the complete dataset.** There is no option to train on
a subset of classes or to stop an epoch early. Those shortcuts existed while
the pipeline was being built and were removed: a run that stops after 3 of 781
batches still prints an epoch summary and writes a checkpoint, which makes it
far too easy to mistake a smoke test for a trained model.
:func:`verify_full_dataset` enforces this at startup and refuses to train on
an incomplete dataset.

Usage::

    # The only mode. Uses CUDA automatically when a GPU is available.
    python -m models.training.train_classifier --epochs 30 --batch-size 256 --device auto

    # Faster on CPU, at some accuracy cost - still all 200 classes, all images.
    python -m models.training.train_classifier --arch resnet18 --no-stem-adapt --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from models.training.augmentation import (
    AugmentationConfig,
    EvalTransform,
    MixCollate,
    TrainTransform,
    mix_criterion,
)
from models.training.dataset import build_dataloaders, find_dataset_root, save_labels


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything that defines a training run.

    Saved alongside the checkpoint so a result can always be reproduced.
    """

    arch: str = "resnet50"
    pretrained: bool = True
    epochs: int = 30
    batch_size: int = 128
    image_size: int = 64

    # --- Optimiser ---------------------------------------------------------
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 5e-2
    # Warmup as a fraction of total steps. 5% is a safe default; raise it if
    # the loss spikes early in training.
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.1

    # --- The three required techniques ------------------------------------
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    scheduler: str = "cosine"  # cosine | onecycle | step | plateau

    # --- Weight averaging ---------------------------------------------------
    # Keep an exponential moving average of the weights alongside the live
    # ones. The averaged weights sit in a flatter region of the loss surface
    # than whatever point the last optimiser step happened to land on, which
    # generalises slightly better - typically a few tenths to about 1.5 points
    # of top-1, for one extra copy of the model in memory and nothing at all
    # at inference time.
    #
    # Off by default: it changes which weights get shipped, and that should be
    # an explicit choice rather than something a default turns on quietly.
    ema: bool = False
    ema_decay: float = 0.9998

    # --- Data --------------------------------------------------------------
    num_workers: int = 4
    adapt_stem: bool = True

    # NOTE: there are deliberately no `class_subset` / `limit_batches` options.
    # Training always runs over the COMPLETE dataset - every class, every
    # image, every batch. Subsetting produced runs that looked finished but
    # had only seen a fraction of the data, and a training script that can
    # quietly train on 3% of the data is a trap. `verify_full_dataset()`
    # enforces this at startup.

    # --- Housekeeping ------------------------------------------------------
    seed: int = 42
    device: str = "auto"
    # Stop early if validation accuracy has not improved for this many epochs.
    early_stopping_patience: int = 0  # 0 disables it; see build_argparser
    # Accumulate gradients over N batches to simulate a larger batch than
    # fits in memory. 1 disables it.
    accumulation_steps: int = 1
    # Resume from the last saved epoch instead of starting over. Essential on
    # a hosted GPU (Colab and friends disconnect), and good practice for any
    # run long enough that losing it hurts.
    resume: bool = True

    # Copy checkpoints and history to a second directory as the run proceeds -
    # typically mounted cloud storage. Resuming only helps if the checkpoint
    # outlives the machine, and on a hosted runtime it does not: the container
    # is recycled and `models/artifacts/` goes with it. Mirroring turns a lost
    # session from "retrain from scratch" into "resume from the last mirror".
    mirror_dir: Path | None = None
    # Full checkpoints are ~96 MB, so they are copied every N epochs rather
    # than every epoch. The history JSON is a few KB and is copied every time,
    # so progress is always visible even between checkpoint mirrors.
    mirror_every: int = 10

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpochResult:
    """Metrics from one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_top1: float
    val_top5: float
    learning_rate: float
    epoch_seconds: float
    grad_norm: float = 0.0
    scaler_scale: float = 0.0
    # Populated only when EMA is on, so a run without it is unchanged.
    ema_top1: float | None = None
    ema_top5: float | None = None
    # Which set of weights produced val_top1: "raw" or "ema".
    selected: str = "raw"


@dataclass
class TrainingHistory:
    """The full record of a run."""

    config: dict[str, Any]
    epochs: list[dict[str, Any]] = field(default_factory=list)
    best_top1: float = 0.0
    best_epoch: int = 0
    total_seconds: float = 0.0
    device: str = "cpu"
    dataset: str = ""
    stopped_early: bool = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ModelEma:
    """An exponential moving average of a model's weights.

    Holds a second copy of the model, updated after every optimiser step as
    ``ema = decay * ema + (1 - decay) * live``. Evaluating the average rather
    than the live weights is a standard, essentially free improvement: SGD and
    AdamW bounce around a minimum rather than settling into it, and the
    average of those positions is closer to the centre than any one of them.

    Two details that are easy to get wrong:

    * **The decay is ramped in.** A fixed 0.9998 from step 1 would leave the
      average dominated by the *initial* weights for thousands of steps, so
      early evaluations would report near-random accuracy and the "best"
      checkpoint logic would be comparing against noise. The warmup schedule
      ``(1 + step) / (10 + step)`` starts near 0.1 and approaches the target,
      so the average is useful from the first epoch.
    * **Buffers are copied, not averaged.** BatchNorm running statistics are
      already running averages; averaging them again would double-smooth them
      and lag the real distribution. Integer buffers (``num_batches_tracked``)
      cannot be averaged at all without silently truncating.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.steps = 0
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    def _current_decay(self) -> float:
        return min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        decay = self._current_decay()

        # Parameters are averaged; buffers are copied. Splitting on
        # parameter-vs-buffer rather than float-vs-int matters: BatchNorm's
        # `running_mean` is a float buffer, so a dtype test silently averages
        # it along with the weights.
        live_params = dict(model.named_parameters())
        for name, value in self.module.named_parameters():
            value.mul_(decay).add_(live_params[name].detach(), alpha=1.0 - decay)

        live_buffers = dict(model.named_buffers())
        for name, value in self.module.named_buffers():
            value.copy_(live_buffers[name])

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "steps": self.steps, "module": self.module.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.steps = int(state.get("steps", 0))
        self.module.load_state_dict(state["module"])


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create the classifier, with its head resized to the class count.

    Tiny-ImageNet's images are 64x64, not the 224x224 ImageNet models expect.
    We feed them at 64x64 anyway rather than upscaling, because upscaling
    quadruples the compute for information that is not there. The consequence
    is that ResNet's aggressive early downsampling (a stride-2 7x7 convolution
    followed by a stride-2 max-pool) reduces a 64x64 input to 16x16 before the
    first residual block, discarding much of the detail. The standard fix,
    applied below, is to replace that stem with a 3x3 stride-1 convolution and
    drop the max-pool — the same adjustment the CIFAR ResNet variants use.
    """
    import torchvision.models as tvm

    if not hasattr(tvm, arch):
        raise ValueError(f"torchvision has no architecture named {arch!r}")

    model = getattr(tvm, arch)(weights="DEFAULT" if pretrained else None)

    if hasattr(model, "fc"):  # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):  # EfficientNet / VGG / DenseNet
        if isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "heads"):  # Vision Transformer
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"do not know how to resize the head of {arch!r}")

    return model


def adapt_stem_for_small_images(model: nn.Module) -> bool:
    """Replace a ResNet's downsampling stem for 64x64 input. See build_model.

    Returns True if the model was adapted.
    """
    if not (hasattr(model, "conv1") and hasattr(model, "maxpool")):
        return False
    old = model.conv1
    if not isinstance(old, nn.Conv2d) or old.kernel_size[0] < 7:
        return False

    model.conv1 = nn.Conv2d(
        old.in_channels, old.out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )
    # The new 3x3 kernel is initialised from the centre of the pretrained 7x7
    # kernel, which preserves the learned colour/edge filters instead of
    # throwing them away and starting from random noise.
    with torch.no_grad():
        model.conv1.weight.copy_(old.weight[:, :, 2:5, 2:5])
    model.maxpool = nn.Identity()
    return True


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig, steps_per_epoch: int
) -> tuple[Any, bool]:
    """Create the learning-rate schedule.

    Returns:
        ``(scheduler, step_per_batch)`` — the flag says whether to call
        ``scheduler.step()`` after every batch or once per epoch. Getting this
        wrong is a classic bug: stepping a per-batch schedule once per epoch
        decays the rate hundreds of times too slowly.
    """
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    if config.scheduler == "cosine":

        def lr_lambda(step: int) -> float:
            """Linear warmup, then a cosine decay to min_learning_rate."""
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            floor = config.min_learning_rate / config.learning_rate
            return floor + (1.0 - floor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), True

    if config.scheduler == "onecycle":
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.learning_rate,
                total_steps=total_steps,
                pct_start=config.warmup_ratio,
            ),
            True,
        )

    if config.scheduler == "step":
        return (
            torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, config.epochs // 3), gamma=0.1
            ),
            False,
        )

    if config.scheduler == "plateau":
        # Reacts to the validation metric rather than following a fixed curve.
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.2, patience=3
            ),
            False,
        )

    raise ValueError(f"unknown scheduler: {config.scheduler!r}")


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Create the optimiser with weight decay applied selectively.

    Weight decay pulls parameters towards zero to discourage over-fitting.
    Applying it to biases and normalisation parameters is actively harmful:
    a BatchNorm scale being pushed to zero switches that channel off. So
    those parameters go in a separate group with decay disabled — a small
    change that is worth a few tenths of a point of accuracy.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


# ---------------------------------------------------------------------------
# Dataset completeness guard
# ---------------------------------------------------------------------------
def verify_full_dataset(data_dir: Path, train_loader: Any, val_loader: Any) -> dict[str, int]:
    """Refuse to train unless the COMPLETE dataset was loaded.

    Plain English:
        This walks the dataset directory, counts what is actually on disk, and
        compares it against what the dataloaders picked up. If a single class
        or image is missing, training stops here rather than producing a
        checkpoint that quietly saw less data than it claims.

    Why it exists: a partially-extracted download, a truncated copy, or a
    subset left in by accident all produce a run that *looks* completely
    normal. It prints an epoch summary, the loss goes down, a checkpoint is
    written. Nothing signals that the model only ever saw a fraction of the
    data. This check turns that silent failure into a loud one.

    Returns:
        Counts of what was verified, for logging.

    Raises:
        RuntimeError: The loaders are not covering everything on disk.
    """
    root = find_dataset_root(data_dir)
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset

    # --- What is on disk? -------------------------------------------------
    train_dir = root / "train"
    disk_classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    disk_train_images = 0
    for wnid in disk_classes:
        images_dir = train_dir / wnid / "images"
        if not images_dir.is_dir():
            images_dir = train_dir / wnid
        disk_train_images += sum(1 for _ in images_dir.glob("*.JPEG"))

    annotations = root / "val" / "val_annotations.txt"
    disk_val_images = sum(
        1
        for line in annotations.read_text(encoding="utf-8").splitlines()
        if len(line.split("	")) >= 2
    )

    # --- What did the loaders actually take? ------------------------------
    loaded_classes = len(getattr(train_ds, "classes", []))
    loaded_train = len(train_ds)
    loaded_val = len(val_ds)

    problems: list[str] = []
    if loaded_classes != len(disk_classes):
        problems.append(
            f"{loaded_classes} of {len(disk_classes)} classes loaded "
            f"({len(disk_classes) - loaded_classes} missing)"
        )
    if loaded_train != disk_train_images:
        problems.append(
            f"{loaded_train:,} of {disk_train_images:,} training images loaded "
            f"({disk_train_images - loaded_train:,} missing)"
        )
    if loaded_val != disk_val_images:
        problems.append(
            f"{loaded_val:,} of {disk_val_images:,} validation images loaded "
            f"({disk_val_images - loaded_val:,} missing)"
        )

    # A degenerate validation split is the ImageFolder bug described in
    # models/training/dataset.py. Catch it before wasting a training run.
    distinct_val_labels = len({label for _, label in getattr(val_ds, "samples", [])})
    if loaded_val and distinct_val_labels <= 1:
        problems.append(
            f"the validation split has only {distinct_val_labels} distinct label(s) - "
            "the labels are not being read from val_annotations.txt"
        )

    if problems:
        detail = "\n  - ".join(problems)
        raise RuntimeError(
            f"Refusing to train on an incomplete dataset:\n  - {detail}\n\n"
            "Training must cover every class and every image. Re-download with:\n"
            "  python scripts/download_datasets.py --dataset tiny_imagenet --data-dir data"
        )

    return {
        "classes": loaded_classes,
        "train_images": loaded_train,
        "val_images": loaded_val,
        "val_labels": distinct_val_labels,
    }


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> list[float]:
    """Top-k accuracy as percentages."""
    maxk = min(max(topk), outputs.size(1))
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.t().eq(targets.view(1, -1).expand_as(pred.t()))

    results = []
    for k in topk:
        k = min(k, maxk)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(float(correct_k.mul_(100.0 / batch_size)))
    return results


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    scheduler: Any,
    step_per_batch: bool,
    config: TrainConfig,
    device: str,
    epoch: int,
    ema: ModelEma | None = None,
) -> tuple[float, float, float]:
    """Run one training epoch.

    Returns:
        ``(mean_loss, mean_grad_norm, current_lr)``.
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    batches = 0

    use_amp = config.mixed_precision and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    # bfloat16 on CPU: it has the same exponent range as float32, so it does
    # not underflow and needs no gradient scaler. float16 would.
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    amp_enabled = config.mixed_precision and (device == "cuda" or torch.cpu.is_available())

    optimizer.zero_grad(set_to_none=True)

    # Every batch, every epoch. There is no early exit.
    for batch_idx, batch in enumerate(loader):
        # MixCollate returns (images, targets_a, targets_b, lam).
        if len(batch) == 4:
            images, targets_a, targets_b, lam = batch
        else:
            images, targets_a = batch
            targets_b, lam = targets_a, 1.0

        images = images.to(device, non_blocking=True)
        targets_a = targets_a.to(device, non_blocking=True)
        targets_b = targets_b.to(device, non_blocking=True)

        # --- MIXED PRECISION: forward pass in 16-bit ----------------------
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(images)
            loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
            # Scale down so that accumulated gradients average rather than sum.
            loss = loss / config.accumulation_steps

        # --- MIXED PRECISION: scale the loss before the backward pass -----
        # Gradients in float16 can underflow to zero. Multiplying the loss by
        # a large factor keeps them representable; the scaler divides them
        # back out before the optimiser step, so the maths is unchanged.
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_step = (batch_idx + 1) % config.accumulation_steps == 0

        if is_step:
            # --- GRADIENT CLIPPING ---------------------------------------
            # Must come AFTER unscaling, or we would be clipping the inflated
            # gradients and the real threshold would be scale-dependent.
            if config.gradient_clip_norm > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                total_grad_norm += float(grad_norm)

            if use_amp:
                # step() is a no-op if the scaler detected inf/NaN gradients;
                # that batch is skipped and the scale factor is reduced.
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # After the step, not before: the average must track the weights
            # the optimiser actually produced.
            if ema is not None:
                ema.update(model)

            # --- LEARNING RATE SCHEDULING --------------------------------
            if step_per_batch and scheduler is not None:
                scheduler.step()

        # .detach() before reading the value: without it we hold a reference to
        # the whole autograd graph for the batch, which leaks memory across an
        # epoch (and torch warns about it).
        batch_loss = loss.detach().item() * config.accumulation_steps
        total_loss += batch_loss
        batches += 1

        if batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch} batch {batch_idx:>5} " f"loss {batch_loss:.4f} lr {lr:.2e}",
                flush=True,
            )

    return (
        total_loss / max(batches, 1),
        total_grad_norm / max(batches, 1),
        optimizer.param_groups[0]["lr"],
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: Any, criterion: nn.Module, device: str, limit: int | None = None
) -> tuple[float, float, float]:
    """Evaluate on the validation set.

    Returns:
        ``(mean_loss, top1_percent, top5_percent)``.
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    seen = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if limit and batch_idx >= limit:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        top1, top5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = targets.size(0)
        total_loss += loss.detach().item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        seen += batch_size

    if seen == 0:
        return 0.0, 0.0, 0.0
    return total_loss / seen, total_top1 / seen, total_top5 / seen


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_top1: float,
    epochs_without_improvement: int,
    ema: ModelEma | None = None,
    history: TrainingHistory,
    config: TrainConfig,
    num_classes: int,
    class_names: list[str],
    adapted: bool,
) -> None:
    """Write a checkpoint that a run can actually be resumed from.

    Saving only the weights is the common mistake. Resuming from weights alone
    restarts the optimiser with no momentum, restarts the learning-rate
    schedule from its warmup, and loses the early-stopping counter — so the
    loss visibly jumps and the remaining schedule is wrong. Everything needed
    to continue exactly where the run stopped is stored here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        # Without this a resumed run would restart the average from the live
        # weights, throwing away every step of smoothing done so far.
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "best_top1": best_top1,
        "epochs_without_improvement": epochs_without_improvement,
        "history": asdict(history),
        # config_as_json, not asdict: a raw `Path` field pickles as a
        # PosixPath, and unpickling a PosixPath on Windows raises outright.
        # A checkpoint trained on a Linux GPU box has to be readable on the
        # laptop it gets pulled back to.
        "config": config_as_json(config),
        "arch": config.arch,
        "num_classes": num_classes,
        "class_names": class_names,
        "stem_adapted": adapted,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    # Write to a temporary file and rename. A process killed midway through a
    # direct write leaves a truncated checkpoint, which is worse than none -
    # the next resume would fail on a file that looks like it should work.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: str,
    ema: ModelEma | None = None,
) -> tuple[int, float, int, list[dict[str, Any]]]:
    """Restore a run from ``path``.

    Returns:
        ``(next_epoch, best_top1, epochs_without_improvement, epoch_history)``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])

    completed = int(ckpt.get("epoch", 0))
    return (
        completed + 1,
        float(ckpt.get("best_top1", 0.0)),
        int(ckpt.get("epochs_without_improvement", 0)),
        list(ckpt.get("history", {}).get("epochs", [])),
    )


def config_as_json(config: TrainConfig) -> dict[str, Any]:
    """``asdict(config)`` with the Path fields turned into strings.

    The history file is JSON and the config is embedded in it verbatim, so a
    single ``Path`` field anywhere in ``TrainConfig`` makes the whole run
    unserialisable - and it fails at the *write*, after the training, which is
    the worst possible moment to discover it. Converting here rather than
    passing ``default=str`` to ``json.dumps`` keeps the failure impossible by
    construction instead of papering over whatever else might be unencodable.
    """
    return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}


def history_path_for(output_dir: Path, config: TrainConfig) -> Path:
    """Where this run's history JSON lives. One definition, several callers."""
    return output_dir / f"{config.arch}_training_history.json"


def mirror_files(paths: list[Path], mirror_dir: Path) -> list[str]:
    """Copy ``paths`` into ``mirror_dir``, reporting what failed rather than raising.

    A mirror is insurance, not part of the run. Cloud-storage mounts go
    unavailable for all the usual reasons - a token expires, the network
    blips, the volume fills - and none of those are a good reason to kill a
    two-hour training job at epoch 47. Failures are returned so the caller can
    print them; the run continues either way.
    """
    import shutil

    problems: list[str] = []
    try:
        mirror_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return [f"{mirror_dir}: {type(exc).__name__}: {exc}"]

    for source in paths:
        if not source.is_file():
            continue
        try:
            # Write to a temporary name in the destination and rename. A copy
            # interrupted partway leaves a truncated file that looks valid,
            # which is worse than no mirror at all.
            tmp = mirror_dir / (source.name + ".partial")
            shutil.copy2(source, tmp)
            tmp.replace(mirror_dir / source.name)
        except Exception as exc:
            problems.append(f"{source.name}: {type(exc).__name__}: {exc}")
    return problems


def train(config: TrainConfig, data_dir: Path, output_dir: Path) -> TrainingHistory:
    """Run the full training loop and save the best checkpoint."""
    torch.manual_seed(config.seed)
    import random

    import numpy as np

    random.seed(config.seed)
    np.random.seed(config.seed)

    device = config.resolve_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device            : {device}")
    if device == "cuda":
        print(f"gpu               : {torch.cuda.get_device_name(0)}")

    # --- Data --------------------------------------------------------------
    aug = AugmentationConfig(image_size=config.image_size, label_smoothing=config.label_smoothing)
    train_loader, val_loader, stats, class_names = build_dataloaders(
        data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_transform=TrainTransform(aug),
        eval_transform=EvalTransform(aug),
        collate_fn=MixCollate(aug),
        # No class_subset: always the full 200-class dataset.
        pin_memory=device == "cuda",
    )
    # Hard gate: every class, every image, or we do not train at all.
    verified = verify_full_dataset(data_dir, train_loader, val_loader)
    print(f"dataset           : {stats.describe()}")
    print(
        f"                    VERIFIED COMPLETE - {verified['classes']} classes, "
        f"{verified['train_images']:,} train images, {verified['val_images']:,} val images "
        f"spanning {verified['val_labels']} val labels"
    )
    print(f"steps per epoch   : {len(train_loader):,} train + {len(val_loader):,} val batches")
    print(f"augmentation      : {json.dumps(aug.describe())}")

    labels_path = save_labels(class_names, output_dir / "tiny_imagenet_labels.json")
    print(f"labels            : {labels_path.name}")

    # --- Model -------------------------------------------------------------
    model = build_model(config.arch, stats.num_classes, config.pretrained)
    adapted = adapt_stem_for_small_images(model) if config.adapt_stem else False
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(
        f"model             : {config.arch}, {params / 1e6:.1f}M params, "
        f"stem adapted for 64px: {adapted}"
    )

    # --- Optimisation ------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(model, config)

    steps_per_epoch = max(1, len(train_loader) // config.accumulation_steps)
    scheduler, step_per_batch = build_scheduler(optimizer, config, steps_per_epoch)

    use_amp = config.mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ema = ModelEma(model, config.ema_decay) if config.ema else None
    if ema is not None:
        print(f"weight averaging  : EMA, decay {config.ema_decay} (warmed in)")

    if config.mirror_dir is not None:
        print(
            f"mirroring to      : {config.mirror_dir} "
            f"(history every epoch, checkpoints every {config.mirror_every})"
        )

    print(
        f"mixed precision   : {'fp16 + GradScaler (cuda)' if use_amp else ('bf16 (cpu)' if config.mixed_precision else 'off')}\n"
        f"gradient clipping : max_norm={config.gradient_clip_norm}\n"
        f"lr schedule       : {config.scheduler}, warmup {config.warmup_ratio:.0%}, "
        f"{config.learning_rate:.1e} -> {config.min_learning_rate:.1e}"
    )

    history = TrainingHistory(
        config=config_as_json(config), device=device, dataset=stats.describe()
    )
    best_top1 = 0.0
    epochs_without_improvement = 0
    start_epoch = 1

    best_path = output_dir / f"{config.arch}_tiny_imagenet_best.pt"
    last_path = output_dir / f"{config.arch}_tiny_imagenet_last.pt"

    # --- Resume ------------------------------------------------------------
    if config.resume and last_path.exists():
        try:
            start_epoch, best_top1, epochs_without_improvement, past = load_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                ema=ema,
            )
            history.epochs = past
            history.best_top1 = best_top1
            print(
                f"resumed           : {last_path.name} - continuing from epoch {start_epoch}"
                f" (best so far {best_top1:.2f}%)"
            )
        except Exception as exc:
            # A corrupt checkpoint must not silently restart a long run as if
            # nothing happened; say so loudly and begin from scratch.
            print(
                f"resume FAILED     : {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print("                    starting from epoch 1", file=sys.stderr)
            start_epoch, best_top1, epochs_without_improvement = 1, 0.0, 0

    if start_epoch > config.epochs:
        print(f"nothing to do: {config.epochs} epochs already completed in {last_path.name}")
        return history

    run_started = time.perf_counter()

    # Written once up front so the per-epoch mirror always has something to
    # copy, and so a run that dies in epoch 1 still leaves its config behind.
    history_path_for(output_dir, config).write_text(
        json.dumps(asdict(history), indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()

        train_loss, grad_norm, lr = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            scheduler,
            step_per_batch,
            config,
            device,
            epoch,
            ema,
        )
        # The whole validation split, every epoch - a partial validation set
        # gives a score that is not comparable between epochs.
        val_loss, raw_top1, raw_top5 = evaluate(model, val_loader, criterion, device)

        # Evaluate the average as well, and ship whichever is better. EMA is
        # usually behind for the first few epochs (it is still catching up
        # from the initial weights) and ahead by the end, so picking one of
        # them up front would either waste the gain or report a worse number
        # than the run actually achieved.
        ema_top1 = ema_top5 = None
        top1, top5, selected = raw_top1, raw_top5, "raw"
        if ema is not None:
            _, ema_top1, ema_top5 = evaluate(ema.module, val_loader, criterion, device)
            if ema_top1 > raw_top1:
                top1, top5, selected = ema_top1, ema_top5, "ema"

        if not step_per_batch and scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(top1)
            else:
                scheduler.step()

        result = EpochResult(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_top1=round(top1, 2),
            val_top5=round(top5, 2),
            learning_rate=lr,
            epoch_seconds=round(time.perf_counter() - epoch_started, 1),
            grad_norm=round(grad_norm, 3),
            scaler_scale=float(scaler.get_scale()) if use_amp else 0.0,
            ema_top1=None if ema_top1 is None else round(ema_top1, 2),
            ema_top5=None if ema_top5 is None else round(ema_top5, 2),
            selected=selected,
        )
        history.epochs.append(asdict(result))

        marker = ""
        if top1 > best_top1:
            best_top1 = top1
            history.best_top1 = round(top1, 2)
            history.best_epoch = epoch
            epochs_without_improvement = 0
            marker = "  <- best"

            torch.save(
                {
                    "model_state_dict": (
                        ema.module.state_dict() if selected == "ema" else model.state_dict()
                    ),
                    "weights": selected,
                    "arch": config.arch,
                    "num_classes": stats.num_classes,
                    "stem_adapted": adapted,
                    "epoch": epoch,
                    "val_top1": top1,
                    "val_top5": top5,
                    "config": config_as_json(config),
                    "class_names": class_names,
                    "saved_at": datetime.now(UTC).isoformat(),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        # Written EVERY epoch, improvement or not. This is the file a resume
        # reads: it carries the optimiser, scheduler and scaler state, so the
        # run continues exactly where it stopped rather than restarting the
        # learning-rate schedule.
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            epoch=epoch,
            best_top1=best_top1,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            config=config,
            num_classes=stats.num_classes,
            class_names=class_names,
            adapted=adapted,
        )

        # Written after the local checkpoint, so the mirror never holds a
        # newer epoch than the file a resume would actually read.
        if config.mirror_dir is not None:
            to_mirror = [history_path_for(output_dir, config)]
            if marker:  # a new best is worth copying immediately
                to_mirror.append(best_path)
            if epoch % config.mirror_every == 0 or epoch == config.epochs:
                to_mirror.append(last_path)
            failures = mirror_files(to_mirror, config.mirror_dir)
            for failure in failures:
                print(f"  mirror failed: {failure}", file=sys.stderr)

        print(
            f"epoch {epoch:>3}/{config.epochs}  "
            f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"top1 {top1:6.2f}%  top5 {top5:6.2f}%  "
            + (f"(raw {raw_top1:5.2f} / ema {ema_top1:5.2f})  " if ema is not None else "")
            + f"lr {lr:.2e}  grad {grad_norm:.2f}  {result.epoch_seconds:.0f}s{marker}",
            flush=True,
        )

        # Guarded by > 0: early stopping is OFF by default, because it is
        # actively wrong for a fixed-length schedule. Cosine decay delivers
        # most of its gain in the final anneal, so validation accuracy
        # legitimately plateaus mid-run while the learning rate is still high.
        # Stopping there discards the part of the schedule that pays.
        if config.early_stopping_patience > 0 and (
            epochs_without_improvement >= config.early_stopping_patience
        ):
            print(f"early stopping: no improvement for {config.early_stopping_patience} epochs")
            history.stopped_early = True
            break

    history.total_seconds = round(time.perf_counter() - run_started, 1)

    history_path = history_path_for(output_dir, config)
    history_path.write_text(json.dumps(asdict(history), indent=2), encoding="utf-8")

    if config.mirror_dir is not None:
        failures = mirror_files([history_path, best_path, last_path], config.mirror_dir)
        for failure in failures:
            print(f"mirror failed: {failure}", file=sys.stderr)
        if not failures:
            print(f"mirrored          : {config.mirror_dir}")

    print(
        f"\nbest top-1        : {history.best_top1:.2f}% at epoch {history.best_epoch}\n"
        f"total time        : {history.total_seconds:.0f}s\n"
        f"checkpoint        : {output_dir / f'{config.arch}_tiny_imagenet_best.pt'}\n"
        f"history           : {history_path}"
    )
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a classifier on Tiny-ImageNet.")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--scheduler", default="cosine", choices=["cosine", "onecycle", "step", "plateau"]
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, dest="gradient_clip_norm")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument(
        "--ema",
        action="store_true",
        help=(
            "Keep an exponential moving average of the weights and ship whichever "
            "of the two scores higher. Costs one extra copy of the model in GPU "
            "memory and nothing at inference time."
        ),
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9998,
        help=(
            "EMA decay. Higher averages over a longer window: 0.9998 spans roughly "
            "the last 5,000 steps. The decay is warmed in, so the average is usable "
            "from the first epoch rather than pinned to the initial weights."
        ),
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-stem-adapt",
        action="store_true",
        help=(
            "Keep the original ImageNet stem. The 64px adaptation costs ~4x compute "
            "because every layer then runs at 4x the spatial resolution; disable it "
            "for fast CPU smoke runs."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        dest="early_stopping_patience",
        help=(
            "Stop after this many epochs without validation improvement. "
            "0 (the default) disables it. Leave it off with cosine or onecycle "
            "scheduling: those are defined over the full epoch count and do "
            "most of their work in the final anneal, so a mid-run plateau is "
            "expected rather than a signal to stop. Useful with --scheduler "
            "plateau or step."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore any existing checkpoint and start from epoch 1. By default a "
            "run continues from the last completed epoch, which is what makes a "
            "long run survive a disconnected hosted GPU session."
        ),
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=None,
        help=(
            "Copy checkpoints and the history JSON here as the run proceeds. "
            "Point it at mounted cloud storage on a hosted runtime: the container "
            "is recycled when the session ends, and without a mirror the run dies "
            "with it."
        ),
    )
    parser.add_argument(
        "--mirror-every",
        type=int,
        default=10,
        help=(
            "Mirror the full resume checkpoint every N epochs. The history JSON "
            "is mirrored every epoch regardless; only the ~96 MB checkpoint is "
            "rate-limited."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models" / "artifacts")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    config = TrainConfig(
        arch=args.arch,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smoothing,
        ema=args.ema,
        ema_decay=args.ema_decay,
        mirror_dir=args.mirror_dir,
        mirror_every=args.mirror_every,
        mixed_precision=not args.no_amp,
        gradient_clip_norm=args.gradient_clip_norm,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        accumulation_steps=args.accumulation_steps,
        resume=not args.no_resume,
        adapt_stem=not args.no_stem_adapt,
    )

    train(config, args.data_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
