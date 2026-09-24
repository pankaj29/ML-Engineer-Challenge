#!/usr/bin/env python
"""Draw the training curves from a run's history JSON.

The notebook plots the same thing inline, but an inline plot has to be
screenshotted to get into the docs, and a screenshot goes stale silently when
the run is repeated. This reads `resnet50_training_history.json` — the file the
training loop writes — so the picture in the documentation is always the
picture of the run whose numbers are quoted beside it.

    python scripts/plot_training_curves.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = REPO_ROOT / "benchmarks" / "reports" / "resnet50_training_history.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "images" / "training-curves.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: pip install matplotlib", file=sys.stderr)
        return 2

    if not args.history.is_file():
        print(f"no history at {args.history}", file=sys.stderr)
        return 2

    history = json.loads(args.history.read_text(encoding="utf-8"))
    epochs = history.get("epochs", [])
    if not epochs:
        print("history contains no epochs", file=sys.stderr)
        return 2

    e = [r["epoch"] for r in epochs]
    has_ema = any(r.get("ema_top1") is not None for r in epochs)
    panels = 4 if has_ema else 3

    fig, ax = plt.subplots(1, panels, figsize=(5.3 * panels, 4))

    ax[0].plot(e, [r["train_loss"] for r in epochs], label="train")
    ax[0].plot(e, [r["val_loss"] for r in epochs], label="validation")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(e, [r["val_top1"] for r in epochs], label="top-1")
    ax[1].plot(e, [r["val_top5"] for r in epochs], label="top-5")
    best = history.get("best_epoch")
    if best:
        ax[1].axvline(best, color="grey", linestyle=":", linewidth=1)
        ax[1].annotate(
            f"best {history.get('best_top1')}%",
            xy=(best, history.get("best_top1", 0)),
            xytext=(4, -14),
            textcoords="offset points",
            fontsize=8,
        )
    ax[1].set(title="Validation accuracy (%)", xlabel="epoch")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].plot(e, [r["learning_rate"] for r in epochs], color="tab:green")
    ax[2].set(title="Learning rate (warmup + cosine)", xlabel="epoch", yscale="log")
    ax[2].grid(alpha=0.3)

    if has_ema:
        # val_top1 is whichever of the two scored higher, so the raw series has
        # to be reconstructed: where EMA was selected, val_top1 IS the EMA
        # figure and the raw one is not stored separately.
        ema = [r.get("ema_top1") for r in epochs]
        selected = [r.get("selected") for r in epochs]
        raw = [
            (r["val_top1"] if s == "raw" else None) for r, s in zip(epochs, selected, strict=True)
        ]
        ax[3].plot(e, ema, label="EMA weights", color="tab:purple")
        raw_e = [x for x, y in zip(e, raw, strict=True) if y is not None]
        raw_v = [y for y in raw if y is not None]
        if raw_v:
            ax[3].scatter(raw_e, raw_v, s=14, color="tab:orange", label="raw weights won", zorder=3)
        ax[3].set(title="Weight averaging", xlabel="epoch", ylabel="top-1 (%)")
        ax[3].legend()
        ax[3].grid(alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110)
    print(f"wrote {args.output}  ({args.output.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(epochs)} epochs, best {history.get('best_top1')}% at {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
