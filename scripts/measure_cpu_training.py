"""Time real training steps on CPU, to show why the classifier was trained on a GPU.

Each configuration runs full optimizer steps (forward, backward, AdamW) on
64x64 inputs, Tiny-ImageNet's native size, with the same model builder the
trainer uses. Epoch time is extrapolated to the 100,000-image training split.

    python scripts/measure_cpu_training.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.training.train_classifier import adapt_stem_for_small_images, build_model

TRAIN_IMAGES = 100_000
BATCH = 32
WARMUP_STEPS = 2
TIMED_STEPS = 6


def measure(arch: str, adapt_stem: bool) -> dict[str, object]:
    model = build_model(arch, num_classes=200, pretrained=False)
    if adapt_stem:
        adapt_stem_for_small_images(model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    images = torch.randn(BATCH, 3, 64, 64)
    labels = torch.randint(0, 200, (BATCH,))

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(images), labels).backward()
        optimizer.step()

    for _ in range(WARMUP_STEPS):
        step()
    started = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    seconds = time.perf_counter() - started
    ips = BATCH * TIMED_STEPS / seconds
    epoch_hours = TRAIN_IMAGES / ips / 3600
    return {
        "arch": arch,
        "stem": "adapted for 64px" if adapt_stem else "original",
        "images_per_second": round(ips, 1),
        "epoch_hours": round(epoch_hours, 2),
        "thirty_epochs_days": round(epoch_hours * 30 / 24, 2),
    }


def main() -> int:
    results = [measure(arch, adapt) for arch in ("resnet50", "resnet18") for adapt in (True, False)]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "input": "64x64, batch 32, fp32, AdamW",
        "timed_steps": TIMED_STEPS,
        "results": results,
    }
    for r in results:
        print(
            f"{r['arch']:<9} {r['stem']:<17} {r['images_per_second']:>7} img/s  "
            f"{r['epoch_hours']:>6} h/epoch  {r['thirty_epochs_days']:>6} days/30 epochs"
        )
    out = REPO_ROOT / "benchmarks" / "reports" / "cpu_training_throughput.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
