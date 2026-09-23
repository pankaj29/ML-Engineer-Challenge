"""Generate the Colab GPU notebook.

The notebook is generated from this script rather than hand-edited, for one
reason: a ``.ipynb`` is JSON with embedded outputs and execution counts, which
makes it miserable to review in a diff and easy to commit with stale results
baked in. Regenerating gives a clean, deterministic file every time::

    python scripts/build_colab_notebook.py

DESIGN: the notebook contains NO machine-learning logic.

Every cell shells out to a module that is already covered by the test suite
(``models.training.train_classifier``, ``models.optimization.*``,
``models.validation.*``). Copying a training loop into a notebook cell would
create a second implementation that drifts from the tested one, and the
notebook version is the one nobody tests. So the notebook is a *driver*: it
prepares the environment, calls the real code, and collects the results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "notebooks" / "colab_gpu_pipeline.ipynb"


def md(text: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """A code cell, with no stored output."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS: list[dict] = [
    md("""
# Multi-Model CV API — GPU Pipeline

Runs the two parts of this project that **need a GPU** and could not be done on
the CPU-only development machine:

1. **Full Tiny-ImageNet fine-tuning** — all 200 classes, 100,000 images, with
   real fp16 mixed precision.
2. **TensorRT export and benchmarking** — never executed before, because
   TensorRT requires an NVIDIA GPU.

It then re-exports, quantizes, benchmarks and validates the trained model, and
packages everything for download.

### How to use it

Works in the **Colab web UI** or through a **VS Code connection to a Colab
runtime** — the setup cell detects which and adapts. Just run the cells in
order.

> **Important:** this notebook contains no training code of its own. Every
> cell calls a module from the repository that is already covered by the test
> suite. That keeps one implementation, not two.

### Expected timings

| GPU | Full 30-epoch run | TensorRT export |
| --- | --- | --- |
| A100 | ~25-40 min | ~5 min |
| L4 / V100 | ~1-1.5 h | ~5 min |
| T4 | ~3-5 h | ~10 min |

Training **resumes automatically** if the session drops — just re-run the
training cell.
"""),
    md(
        "## 1. Check the GPU\n\nConfirm what hardware was allocated before committing to a long run."
    ),
    code("""
import subprocess, sys

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "NO GPU DETECTED")

try:
    import torch
    print(f"torch      : {torch.__version__}")
    print(f"CUDA avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU        : {p.name}  ({p.total_memory / 1e9:.0f} GB, SM {p.major}.{p.minor})")
    else:
        print("\\nNo GPU. In Colab: Runtime -> Change runtime type -> GPU.")
        print("Everything below still runs on CPU, just far more slowly.")
except ImportError:
    print("torch not installed yet - the setup cell below installs it.")
"""),
    md("""
## 2. Get the code

Clones the repository from GitHub, so the runtime's copy always matches the
last push. A fix made locally reaches the GPU as soon as you push it — no
zipping, no uploading, and no stale copy of a script silently reintroducing a
bug that was already fixed.

The repository is public, so this needs no token, no credentials and no
prompt.

1. **An existing checkout** — fetched and **hard-reset to the remote**, so it
   cannot be stale. It prints the commit it moved from and to.
2. **`git clone`** — if there is no checkout yet.

A directory that is a checkout of some *other* repository (your own local
working copy, say) is left untouched.

> **Why reset rather than reuse.** Reusing a directory because it merely
> exists is how a runtime ends up running code from before the last push. That
> happened repeatedly here, and each time it cost a training run at stale
> settings before anyone noticed.

> **Why nothing here prompts.** `files.upload()` and `getpass()` both render a
> *browser* widget. Driving a Colab kernel from VS Code means there is no
> browser session to render into, so they hang forever with *"Upload widget is
> only available when the cell has been executed in the current browser
> session."* This notebook never calls either.
"""),
    code("""
import os, subprocess, sys
from pathlib import Path

REPO_NAME = "ML-Engineer-Challenge"
REPO_URL = "https://github.com/pankaj29/ML-Engineer-Challenge.git"
BRANCH = "main"
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")


def looks_like_repo(path):
    return (Path(path) / "models" / "training" / "train_classifier.py").is_file()


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=None if cwd is None else str(cwd), capture_output=True, text=True
    )


def _purge_stale_modules():
    # Updating files on disk does NOT update an already-running kernel.
    # Python caches imported modules in sys.modules, so `from
    # models.optimization... import NewThing` keeps returning the OLD module
    # and fails with ImportError even though the file on disk is correct.
    #
    # Dropping this project's packages from the cache makes the next import
    # read the new files. Third-party modules are left alone - reloading torch
    # or tensorrt mid-session is how you get two incompatible copies of a C
    # extension in one process.
    ours = ("models", "api", "worker", "db", "scripts")
    stale = [m for m in sys.modules if m.split(".")[0] in ours]
    for name in stale:
        del sys.modules[name]
    if stale:
        print("reloaded modules     : " + str(len(stale)) + " (kernel had cached the old code)")
    else:
        print("reloaded modules     : none cached yet")


def is_our_checkout(path):
    # True if `path` is a git clone whose origin is this repository.
    done = git("-C", str(path), "remote", "get-url", "origin")
    return done.returncode == 0 and REPO_NAME.lower() in done.stdout.strip().lower()


REPO = None

# --- 1. An existing checkout: UPDATE it, never just trust it ---------------
#
# Reusing a directory because it merely exists is how a runtime ends up
# silently running code from before the last push - which has happened
# repeatedly, costing whole training runs at stale settings. If the directory
# is a checkout of this repo, it gets forced to match the remote.
for candidate in [Path.cwd(), Path.cwd().parent, Path("/content") / REPO_NAME]:
    if not looks_like_repo(candidate):
        continue
    if not is_our_checkout(candidate):
        # A local working copy that is not a clone of this repo - someone's own
        # checkout. Leave it entirely alone.
        REPO = candidate.resolve()
        print("found repo in place  : " + str(REPO) + " (not a clone of this repo; left as-is)")
        break

    before = git("-C", str(candidate), "rev-parse", "--short", "HEAD").stdout.strip()
    fetched = git("-C", str(candidate), "fetch", "--depth", "1", "origin", BRANCH)
    if fetched.returncode != 0:
        print(fetched.stderr.strip(), file=sys.stderr)
        print("could not reach the remote; using the checkout as-is", file=sys.stderr)
        REPO = candidate.resolve()
        break

    # reset --hard, not pull: a shallow clone cannot always fast-forward, and
    # this cell's job is "make the runtime match the remote", not "merge".
    git("-C", str(candidate), "reset", "--hard", "origin/" + BRANCH)
    after = git("-C", str(candidate), "rev-parse", "--short", "HEAD").stdout.strip()
    REPO = candidate.resolve()
    if before == after:
        print("repo up to date      : " + str(REPO) + " @ " + after)
    else:
        print("repo UPDATED         : " + str(REPO) + "  " + before + " -> " + after)
    # Unconditional, NOT only when the commit moved. "Already at the right
    # commit" does not mean the kernel is running that code: an earlier run of
    # this cell may have updated the files while the kernel had already
    # imported the previous version. Purging is cheap and touches only this
    # project's modules, so there is no reason to make it conditional.
    _purge_stale_modules()
    break

# --- 2. No checkout yet: clone -------------------------------------------
if REPO is None:
    target = (Path("/content") if IN_COLAB else Path.cwd()) / REPO_NAME
    print("cloning              : " + REPO_URL)
    done = git("clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(target))
    if done.returncode == 0 and looks_like_repo(target):
        REPO = target.resolve()
        head = git("-C", str(REPO), "rev-parse", "--short", "HEAD").stdout.strip()
        print("cloned to            : " + str(REPO) + " @ " + head)
    else:
        print(done.stderr.strip(), file=sys.stderr)
        print()
        print("Clone failed. The repository is public and needs no credentials,")
        print("so this is almost always one of:")
        print("  - this runtime has no network access to github.com")
        print("  - the repository was moved, renamed, or made private again")
        raise SystemExit("repository not available")

os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("working directory    : " + str(Path.cwd()))

# The dataset lives OUTSIDE the repository, and DATA_DIR is defined here
# rather than in the download cell because later cells (quantisation
# calibration, validation) need it too. Defining it at the point the working
# directory is settled means any cell can be run after a kernel restart
# without a NameError.
#
# Why outside the repo: on a hosted runtime the checkout is disposable -
# re-cloning is the normal way to pick up a fix - and a 240 MB dataset inside
# the working tree gets deleted with it every time.
DATA_DIR = Path("/content/data") if Path("/content").exists() else Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("data directory       : " + str(DATA_DIR))

# --- The training config, in one place -------------------------------------
# The next cell reads these, so there is exactly one definition of what is
# being trained. Change it here, not in the command below.
ARCH = "resnet50"
EPOCHS = 60
IMAGE_SIZE = 128
STEM_ADAPTED = False      # False = keep ResNet's original pretrained stem
BATCH_SIZE = 256
# LR 3e-4, not 1e-3. AdamW at 1e-3 suits a network with a randomly
# initialised stem (the 64px adapted-stem config), where part of the model
# trains from scratch. With STEM_ADAPTED = False the whole pretrained
# ResNet-50 is intact, and 1e-3 erodes exactly the features 128px was chosen
# to preserve: validation top-1 regressed 70.5% -> 64.8% while train loss
# kept falling, with non-finite gradients appearing.
#
# PATIENCE 15, not the script default of 0 (off) and not the 8 that burned
# three runs. A cosine schedule does most of its work in the final anneal, so
# validation accuracy plateaus mid-run as a matter of course - 8 epochs of
# flatness is normal there, not a signal. 15 is long enough to sit through
# that and short enough to cut a genuinely dead run. It is still a judgement
# call: if it fires before roughly epoch 45, suspect the plateau rather than
# the model, and re-run with PATIENCE = 0 to disable it.
LR = "3e-4"            # see the note below before changing this
PATIENCE = 15          # stop after this many epochs with no val improvement
"""),
    md("""
## 3. Install dependencies

Installed from the project's **`requirements-train.txt` and
`requirements-gpu.txt`** — the same pins the test suite ran against — so this
GPU run cannot drift from the versions everything else was verified with.

Three packages are deliberately *not* installed, because this runtime already
provides them correctly:

| Skipped | Why |
| --- | --- |
| `torch`, `torchvision` | The runtime's build is matched to its CUDA version and driver. Installing our `torch~=2.9.0` pin would **downgrade it to a CPU-only wheel** and silently cost you the A100. |
| `numpy` | Shares a C ABI with torch; the runtime's version is matched to its build. |
| `onnxruntime` | The CPU package — `onnxruntime-gpu` replaces it. |

Serving-only dependencies (FastAPI, Redis, Celery, the Postgres drivers) are
skipped too: this machine trains and exports, it does not serve. Pass `--all`
to include them.

The script then **verifies torch can still see the GPU**, and fails loudly if
the install broke CUDA — which is the exact failure it exists to prevent.

Preview the plan without installing anything:
`python scripts/install_for_gpu_runtime.py --dry-run`
"""),
    code("""
# Everything comes from the requirements files; the script only filters out
# what the runtime already provides and the serving-only packages this
# machine does not need.
#
# --with-tensorrt pulls in the extras that cell 7 requires.
!python scripts/install_for_gpu_runtime.py --with-tensorrt
"""),
    md("""
## 4. Download Tiny-ImageNet

~240 MB, a minute or two on Colab's connection. Skipped if already present
(for example when the repo lives on Drive).
"""),
    code("""
from pathlib import Path

# DATA_DIR comes from the "Get the code" cell above.
print("data directory    : " + str(DATA_DIR))

if (DATA_DIR / "tiny-imagenet-200").exists():
    print("dataset already present - not re-downloading")
else:
    !python scripts/download_datasets.py --dataset tiny_imagenet --data-dir {DATA_DIR}

# Verify it is COMPLETE before committing to a long run. The training script
# refuses to start on a partial dataset anyway, but failing here is cheaper.
from models.training.dataset import TinyImageNetTrain, TinyImageNetVal, find_dataset_root

root = find_dataset_root(DATA_DIR)
train = TinyImageNetTrain(root)
val = TinyImageNetVal(root, train.class_to_idx)
distinct = len({label for _, label in val.samples})

print("root      : " + str(root))
print("classes   : " + str(len(train.classes)))
print("train     : " + format(len(train), ",") + " images")
print("val       : " + format(len(val), ",") + " images across " + str(distinct) + " labels")
assert len(train.classes) == 200 and len(train) == 100_000 and distinct == 200, (
    "dataset is incomplete - re-download before training"
)
print()
print("Dataset verified COMPLETE.")
"""),
    md("""
## 5. Recover a previous checkpoint

Training writes `models/checkpoints/last.pth` every epoch, and the training
script resumes from it automatically. But that file lives on the runtime's
local disk, which does not survive a reset — and a fresh `git clone` lands in
a new directory that has no checkpoints in it.

So before training, this looks for a usable checkpoint elsewhere on the
machine and adopts the most recent one. Without it, a dropped session or a
re-clone silently costs a full training run.

It never overwrites a newer checkpoint already in place, and it is safe to
re-run. If nothing is found it says so and training simply starts from
scratch.
"""),
    code("""
import shutil, time
from pathlib import Path

# These names come from train_classifier.py (see best_path / last_path there).
# Checkpoints live in models/artifacts/ alongside the ONNX exports - NOT in a
# checkpoints/ directory - so anything that moves them must name the files
# individually rather than move the folder.
ART_DIR = Path.cwd() / "models" / "artifacts"
LAST_NAME = ARCH + "_tiny_imagenet_last.pt"
BEST_NAME = ARCH + "_tiny_imagenet_best.pt"
HIST_NAME = ARCH + "_training_history.json"
CKPT_FILES = [LAST_NAME, BEST_NAME, HIST_NAME]

mine = ART_DIR / LAST_NAME

# --- Adopt a checkpoint from a previous session, if there is one -----------
search_roots = [Path("/content"), Path("/content/drive/MyDrive")]
candidates = []
for root in search_roots:
    if not root.exists():
        continue
    for found in root.glob("**/models/artifacts/" + LAST_NAME):
        if found.resolve() != mine.resolve() and found.is_file():
            candidates.append(found)

if not candidates:
    print("no checkpoint found in another directory")
else:
    newest = max(candidates, key=lambda f: f.stat().st_mtime)
    for found in sorted(candidates, key=lambda f: f.stat().st_mtime, reverse=True):
        marker = " <- newest" if found == newest else ""
        print("found: " + str(found) + marker)

    if mine.exists() and mine.stat().st_mtime >= newest.stat().st_mtime:
        print()
        print("keeping the checkpoint already in place: " + str(mine))
    else:
        ART_DIR.mkdir(parents=True, exist_ok=True)
        # best.pt and the history belong with last.pt: resuming without the
        # history loses the training curves.
        for name in CKPT_FILES:
            src = newest.parent / name
            if src.is_file():
                shutil.copy2(src, ART_DIR / name)
        print()
        print("adopted: " + str(newest))
        print("     -> " + str(ART_DIR))

# --- Is that checkpoint resumable into THIS config? ------------------------
#
# Resuming only works if the architecture matches. A 64px adapted-stem
# checkpoint has a different conv1 shape from a 128px original-stem model, so
# load_state_dict would fail - or worse, a partial load could succeed and
# train something subtly wrong.
#
# Incompatible checkpoints are MOVED ASIDE, never deleted: a finished run is
# expensive, and "superseded" is not "worthless".
if not mine.exists():
    print()
    print("no checkpoint in place; the next cell will train from epoch 1")
else:
    import torch

    state = torch.load(mine, map_location="cpu", weights_only=False)
    ckpt_cfg = state.get("config") or {}
    ckpt_arch = state.get("arch", ckpt_cfg.get("arch"))
    ckpt_size = ckpt_cfg.get("image_size")
    ckpt_stem = state.get("stem_adapted", ckpt_cfg.get("adapt_stem"))

    mismatches = []
    if ckpt_arch is not None and ckpt_arch != ARCH:
        mismatches.append("arch: checkpoint " + str(ckpt_arch) + ", wanted " + ARCH)
    if ckpt_size is not None and ckpt_size != IMAGE_SIZE:
        mismatches.append(
            "image_size: checkpoint " + str(ckpt_size) + ", wanted " + str(IMAGE_SIZE)
        )
    if ckpt_stem is not None and bool(ckpt_stem) != STEM_ADAPTED:
        mismatches.append(
            "stem_adapted: checkpoint " + str(bool(ckpt_stem)) + ", wanted " + str(STEM_ADAPTED)
        )

    epoch = state.get("epoch", "?")
    best = state.get("best_top1")
    best_str = ("%.2f%%" % best) if isinstance(best, (int, float)) else "n/a"
    print()
    print("checkpoint: epoch " + str(epoch) + ", best top-1 " + best_str)

    if mismatches:
        stash = ART_DIR / ("superseded-" + time.strftime("%Y%m%d-%H%M%S"))
        stash.mkdir(parents=True, exist_ok=True)
        print()
        print("INCOMPATIBLE with the config above:")
        for line in mismatches:
            print("  - " + line)
        # Move the checkpoint files only. models/artifacts/ also holds the
        # ONNX exports, which must stay where they are.
        for name in CKPT_FILES:
            src = ART_DIR / name
            if src.is_file():
                shutil.move(str(src), str(stash / name))
        print()
        print("moved aside (not deleted): " + str(stash))
        print("the next cell will train from epoch 1")
    else:
        print("compatible - the next cell will resume from here")
        print()
        print("NOTE: resuming also restores the no-improvement counter. If a")
        print("previous run stopped early, re-running may stop again straight")
        print("away. Set EPOCHS above and rerun with a fresh start if so.")
"""),
    md("""
## 6. Train on the full dataset

**All 200 classes, all 100,000 images, every batch.** The script has no option
to subset — `verify_full_dataset()` refuses to start otherwise.

Demonstrates the three techniques the brief requires:

* **Mixed precision** — fp16 autocast + `GradScaler` (real fp16 here, unlike
  the CPU's bf16 fallback)
* **Gradient clipping** — `clip_grad_norm_` after unscaling
* **LR scheduling** — cosine decay with linear warmup

**Why 128px with the original stem.** Tiny-ImageNet images are 64x64, so
this upsamples them. That is deliberate. ResNet-50's stem is a stride-2 7x7
convolution followed by a stride-2 maxpool, which reduces its input 4x before
the first residual block. Feed it 64px and `layer1` sees 16x16 - too little
spatial detail - so the usual fix is to replace the stem with a stride-1 3x3
and drop the maxpool. That works, but it throws away pretrained stem weights
and leaves `layer1` running at 64x64.

Feeding 128px through the **original** stem gives `layer1` a 32x32 map and
uses ResNet-50 exactly as it was pretrained. It is the better transfer setup,
and - counter-intuitively - the *cheaper* one: 32x32 into `layer1` is a
quarter the spatial area of the 64px-with-adapted-stem config, so a 128px
epoch runs faster than a 64px one did, not slower.

**On batch size.** 256 at 128px with the original stem leaves comfortable
headroom on a 40 GB A100 - noticeably more than the adapted-stem config used,
for the reason above. Raise it to 512 to push throughput; drop to 128 on a
T4 (16 GB).

**On epoch count.** 60, not 30. The augmentation pipeline here is heavy
(RandAugment + MixUp + CutMix), and heavy regularisation needs a long schedule
to pay for itself - at 30 epochs it can cost accuracy rather than buy it. Note
that the cosine schedule is defined over the *total* epoch count, so changing
this number changes the whole LR curve: it is a fresh run (`--no-resume`), not
an extension of a shorter one.

> **Resume is on by default.** If the session drops, just re-run this cell —
> it continues from the last completed epoch with the optimiser and LR
> schedule intact. Pass `--no-resume` to force a fresh start.
"""),
    code("""
!python -u -m models.training.train_classifier \\
    --arch {ARCH} \\
    --epochs {EPOCHS} \\
    --image-size {IMAGE_SIZE} \\
    --batch-size {BATCH_SIZE} \\
    --lr {LR} \\
    --num-workers 8 \\
    --scheduler cosine \\
    --warmup-ratio 0.05 \\
    --grad-clip 1.0 \\
    --label-smoothing 0.1 \\
    --patience {PATIENCE} \\
    --no-stem-adapt \\
    --data-dir {DATA_DIR} \\
    --device cuda
"""),
    md("### Training curves"),
    code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt

history = json.loads(Path("models/artifacts/resnet50_training_history.json").read_text())
epochs = history["epochs"]

if epochs:
    e = [r["epoch"] for r in epochs]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(e, [r["train_loss"] for r in epochs], label="train")
    ax[0].plot(e, [r["val_loss"] for r in epochs], label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(e, [r["val_top1"] for r in epochs], label="top-1")
    ax[1].plot(e, [r["val_top5"] for r in epochs], label="top-5")
    ax[1].set(title="Validation accuracy (%)", xlabel="epoch"); ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(e, [r["learning_rate"] for r in epochs], color="tab:green")
    ax[2].set(title="Learning rate (warmup + cosine)", xlabel="epoch", yscale="log")
    ax[2].grid(alpha=.3)

    plt.tight_layout(); plt.show()
    print(f"best top-1: {history['best_top1']}% at epoch {history['best_epoch']}")
    print(f"total time: {history['total_seconds'] / 60:.1f} min on {history['device']}")
else:
    print("no epochs recorded yet")
"""),
    md("""
## 7. Export the fine-tuned model

Exports to ONNX with **numerical verification** against PyTorch, then applies
INT8 quantization calibrated on real images.
"""),
    code("""
import json
from pathlib import Path

import torch

from models.optimization.export_onnx import export_to_onnx
from models.training.train_classifier import build_model, adapt_stem_for_small_images

ckpt = torch.load("models/artifacts/resnet50_tiny_imagenet_best.pt",
                  map_location="cpu", weights_only=False)
print(f"checkpoint: epoch {ckpt['epoch']}, top-1 {ckpt['val_top1']:.2f}%")

model = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False)
if ckpt.get("stem_adapted"):
    adapt_stem_for_small_images(model)
model.load_state_dict(ckpt["model_state_dict"])

result = export_to_onnx(
    model,
    Path("models/artifacts/resnet50-tiny-imagenet.onnx"),
    input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
    name="resnet50-tiny-imagenet",
)
print(f"onnx      : {result.size_mb:.1f} MB")
print(f"max diff  : {result.max_abs_diff:.2e}  verified={result.verified}")
for note in result.notes:
    print(f"note      : {note}")

# Save the class names alongside the model, so serving returns readable labels.
Path("models/artifacts/tiny_imagenet_labels.json").write_text(
    json.dumps(ckpt["class_names"]), encoding="utf-8"
)
print(f"labels    : {len(ckpt['class_names'])} classes")
"""),
    code("""
from pathlib import Path

from api.utils.image_processing import PreprocessConfig
from models.optimization.quantize import quantize_onnx_static

# Static QDQ, calibrated on real images. Dynamic quantization measured 13x
# SLOWER than fp32 on CPU - see docs/TECHNICAL.md.
q = quantize_onnx_static(
    Path("models/artifacts/resnet50-tiny-imagenet.onnx"),
    DATA_DIR / "tiny-imagenet-200" / "tiny-imagenet-200" / "val" / "images",
    PreprocessConfig(size=(IMAGE_SIZE, IMAGE_SIZE)),
    num_calibration=200,
)
print(q.summary())
"""),
    md("""
## 8. TensorRT

**This is the part that has never run.** TensorRT compiles the ONNX graph for
this specific GPU: it fuses layers, picks the fastest kernel for each operation
by timing candidates, and runs in fp16.

Engines are **not portable** — built for one GPU architecture and one TensorRT
version. Build on the machine that will serve.
"""),
    code("""
from models.optimization.export_tensorrt import tensorrt_available

available, reason = tensorrt_available()
print(f"TensorRT available: {available}  ({reason})")

if not available and "tensorrt package" in reason:
    !pip install -q tensorrt pycuda 2>&1 | tail -2
    import importlib, models.optimization.export_tensorrt as trt_mod
    importlib.reload(trt_mod)
    available, reason = trt_mod.tensorrt_available()
    print(f"after install     : {available}  ({reason})")
"""),
    code("""
from pathlib import Path

from models.optimization.export_tensorrt import (
    UnsupportedPrecisionError,
    benchmark_engine,
    build_engine,
    tensorrt_available,
)

available, reason = tensorrt_available()
if not available:
    print(f"SKIPPED: {reason}")
else:
    for precision in ("fp16", "fp32"):
        print(f"\\n--- building {precision} engine ---")
        try:
            res = build_engine(
                Path("models/artifacts/resnet50-tiny-imagenet.onnx"),
                precision=precision,
                max_batch_size=32,
                workspace_gb=8.0,
            )
            print(res.summary())
            for note in res.notes:
                print(f"  note: {note}")

            bench = benchmark_engine(
                Path(res.engine_path),
                input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                iterations=200,
                warmup=50,
            )
            print(f"  latency: p50 {bench['p50_ms']:.3f} ms | p95 {bench['p95_ms']:.3f} ms "
                  f"| {bench['throughput_ips']:.0f} img/s")
        except UnsupportedPrecisionError as exc:
            # Not a failure: this TensorRT build cannot express the precision
            # without an ONNX file already in it. Recorded as a skip so the
            # run reads honestly.
            print("  SKIPPED: " + str(exc))
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
"""),
    md("""
## 9. Benchmark every format on this GPU

The first GPU numbers for this project. Compare against the CPU baselines in
`benchmarks/reports/BENCHMARKS.md`.
"""),
    code("""
from pathlib import Path

from models.optimization.benchmark import benchmark_onnx, environment_info, render_markdown

results = []
for name in ("resnet50-tiny-imagenet.onnx", "resnet50-tiny-imagenet_int8_static.onnx"):
    path = Path("models/artifacts") / name
    if not path.exists():
        continue
    print(f"benchmarking {name} ...")
    results.extend(
        benchmark_onnx(path, input_shape=(3, IMAGE_SIZE, IMAGE_SIZE), batch_sizes=(1, 8, 32),
                       iterations=100, warmup=20, device="cuda")
    )

print()
for r in sorted(results, key=lambda x: (x.name, x.batch_size)):
    print(r.summary())

# Name the report after the device actually used, not the one requested.
# ONNX Runtime falls back to CPU without error when the CUDA provider is
# missing, and a file called BENCHMARKS_GPU.md full of CPU numbers is a
# misleading artefact that outlives the session that produced it.
devices = {r.device for r in results}
if devices == {"cuda"}:
    out = Path("benchmarks/reports/BENCHMARKS_GPU.md")
else:
    out = Path("benchmarks/reports/BENCHMARKS_GPU_CPU_FALLBACK.md")
    print()
    print("WARNING: ONNX Runtime ran on " + ", ".join(sorted(devices)) + ", not cuda.")
    print("These are NOT GPU numbers. The TensorRT results above are the only")
    print("true GPU figures in this run. Check that onnxruntime-gpu is installed")
    print("and that plain onnxruntime is not shadowing it.")

env = environment_info()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(render_markdown(results, env), encoding="utf-8")
print(f"\\nwrote {out}")
"""),
    md(
        "## 10. Validate the trained model\n\nThe same gate the CPU models pass: determinism, batch invariance, output sanity, calibration and latency."
    ),
    code("""
from pathlib import Path

from models.registry import Registry
from models.validation.validate import load_eval_samples, validate_model

Registry().register(
    name="resnet50-tiny-imagenet",
    version="1.0.0",
    task="classification",
    artifacts={"onnx": "resnet50-tiny-imagenet.onnx"},
    preprocess="tiny_imagenet",
    labels_file="tiny_imagenet_labels.json",
    num_classes=200,
    input_shape=[1, 3, IMAGE_SIZE, IMAGE_SIZE],
    description="ResNet-50 fine-tuned on Tiny-ImageNet (200 classes) with AMP, "
                "gradient clipping and cosine LR scheduling.",
    overwrite=True,
)

# NOW accuracy is measurable: the model and the evaluation set share a label
# space (200 classes), which was not true of the ImageNet-1k model.
samples = load_eval_samples(DATA_DIR, limit=2000)
report = validate_model("resnet50-tiny-imagenet:1.0.0", eval_samples=samples, max_p95_ms=1000)

print(report.summary())
for check in report.checks:
    mark = " ok " if check["passed"] else ("FAIL" if check["severity"] == "critical" else "warn")
    print(f"  [{mark}] {check['name']:<20} {check['message']}")

print("\\nmetrics:")
for k, v in sorted(report.metrics.items()):
    print(f"  {k:<28} {v}")
"""),
    md(
        "## 11. Download the results\n\nBrings the trained weights, exports and reports back to your machine."
    ),
    code("""
import shutil, sys
from pathlib import Path

bundle = Path("/content/gpu_results") if Path("/content").exists() else Path("gpu_results")
if bundle.exists():
    shutil.rmtree(bundle)
(bundle / "artifacts").mkdir(parents=True)
(bundle / "reports").mkdir(parents=True)

wanted = [
    "resnet50_tiny_imagenet_best.pt",
    "resnet50_training_history.json",
    "resnet50-tiny-imagenet.onnx",
    "resnet50-tiny-imagenet_int8_static.onnx",
    "tiny_imagenet_labels.json",
]
for name in wanted:
    src = Path("models/artifacts") / name
    if src.exists():
        shutil.copy2(src, bundle / "artifacts" / name)
        print(f"  + {name}  ({src.stat().st_size / 1e6:.1f} MB)")

for engine in Path("models/artifacts").glob("*.engine"):
    shutil.copy2(engine, bundle / "artifacts" / engine.name)
    print(f"  + {engine.name}")

for report in Path("benchmarks/reports").glob("*"):
    if report.is_file():
        shutil.copy2(report, bundle / "reports" / report.name)

shutil.copy2("models/registry.json", bundle / "registry.json")

archive = shutil.make_archive(str(bundle), "zip", bundle)
size = Path(archive).stat().st_size / 1e6
print(f"\\nbundle: {archive}  ({size:.1f} MB)")

if "google.colab" in sys.modules:
    from google.colab import files
    files.download(archive)
else:
    print("Not in Colab - copy the file above off the machine yourself.")
"""),
    md("""
## Next steps, back on your machine

```bash
# 1. Unzip into the repo
unzip gpu_results.zip -d /tmp/gpu && \\
  cp /tmp/gpu/artifacts/* models/artifacts/ && \\
  cp /tmp/gpu/reports/*  benchmarks/reports/

# 2. Register the fine-tuned model as the classification default
python -m models.registry register \\
    --name resnet50-tiny-imagenet --version 1.0.0 --task classification \\
    --onnx resnet50-tiny-imagenet.onnx --labels tiny_imagenet_labels.json \\
    --preprocess tiny_imagenet --num-classes 200 \\
    --input-shape 1,3,128,128 --default --overwrite

# 3. Confirm it serves
python -m models.registry validate
docker compose restart ml-api
curl -X POST http://localhost/api/v1/classify \\
     -H "X-API-Key: dev-key-pro" -H "Content-Type: application/json" \\
     -d "{\\"image_base64\\": \\"$(base64 -w0 photo.jpg)\\"}"
```

Then update `docs/ASSUMPTIONS.md` §2.1 and §2.2 — both gaps are now closed —
and fold the GPU numbers into `docs/TECHNICAL.md`.
"""),
]


def _notebook() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(_notebook(), indent=1), encoding="utf-8")
    return OUTPUT


def check() -> int:
    """Report whether the notebook on disk matches what this script produces.

    The notebook is a BUILD ARTEFACT. Opening it in an editor and saving -
    which Jupyter and VS Code do on their own, to record execution outputs -
    silently overwrites generated content. That has happened repeatedly, and
    each time the stale copy looked fine until something failed on a GPU
    runtime minutes into a run.

    Run with --check in CI so a stale notebook fails the build instead of
    being discovered the expensive way.
    """
    if not OUTPUT.exists():
        print(f"MISSING: {OUTPUT}", file=sys.stderr)
        return 1

    on_disk = json.loads(OUTPUT.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(_notebook()))

    # Compare only the source of each cell. Execution counts and outputs are
    # expected to differ - that is what running the notebook does - and are
    # not what this guard is protecting.
    def sources(nb):
        return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]

    if sources(on_disk) == sources(expected):
        print(f"up to date: {OUTPUT}")
        return 0

    disk, exp = sources(on_disk), sources(expected)
    print(f"STALE: {OUTPUT} does not match {Path(__file__).name}", file=sys.stderr)
    if len(disk) != len(exp):
        print(f"  cell count: on disk {len(disk)}, expected {len(exp)}", file=sys.stderr)
    for i, (a, b) in enumerate(zip(disk, exp, strict=False), start=1):
        if a != b:
            print(f"  cell {i} ({a[0]}) differs", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"Fix: python {Path(__file__).as_posix()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(check())
    path = build()
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {path}")
    print(f"  {len(CELLS)} cells ({n_code} code, {len(CELLS) - n_code} markdown)")
