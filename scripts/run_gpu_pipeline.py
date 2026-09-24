#!/usr/bin/env python
"""One-shot GPU pipeline: clone, install, train, export, optimise, benchmark, validate.

Run this instead of stepping through ``notebooks/colab_gpu_pipeline.ipynb``.
It does the same work with no cell-by-cell interaction, and - the reason it
exists - it saves results to Drive continuously rather than only at the end.

In Colab, one cell:

    !curl -sSL https://raw.githubusercontent.com/pankaj29/ML-Engineer-Challenge/main/scripts/run_gpu_pipeline.py -o /content/run.py
    !python -u /content/run.py

Nothing else. The script clones the repository itself, so it can be fetched
and run before the checkout exists.

Design notes, all of them learned the hard way on this project:

* **Drive is mounted first, and a failure is fatal.** The previous design
  mounted at the very end, so an expired token surfaced only after two hours
  of training - with the results on a container about to be recycled. If the
  run cannot save, there is no point starting it. ``--no-drive`` opts out
  deliberately.
* **Checkpoints are mirrored to Drive as the run proceeds**, not copied at the
  end. Resuming only helps if the checkpoint outlives the machine.
* **A resumable run is adopted from Drive automatically**, and refused if its
  configuration does not match the one requested. Resuming a 128px checkpoint
  into a 224px run produces a plausible-looking result that means nothing.
* **Every stage delivers what it produced before the next one starts.** A
  failure in TensorRT must not cost you the trained model.
* **Optional stages fail soft, required stages fail hard.** No GPU means no
  TensorRT, which is a fact about the machine, not a broken run. A failed
  export, however, stops everything after it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration - change it here, nowhere else
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/pankaj29/ML-Engineer-Challenge.git"
REPO_NAME = "ML-Engineer-Challenge"
BRANCH = "main"

ARCH = "resnet50"
EPOCHS = 60
IMAGE_SIZE = 224
BATCH_SIZE = 256
LEARNING_RATE = "3e-4"
NUM_WORKERS = 8

# PATIENCE 0 - early stopping OFF, and this is not a casual default.
#
# A cosine schedule spends its first two thirds at a high learning rate and
# delivers most of its accuracy in the final anneal. Validation accuracy
# therefore plateaus mid-run as a matter of course. A patience of 15 killed a
# 60-epoch run at epoch 32, having peaked at epoch 17 with the learning rate
# still at 86% of maximum - the run never reached the part that pays. Two
# earlier runs died the same way at a patience of 8.
#
# Early stopping is the right tool for `--scheduler plateau` or `step`. With a
# fixed-length schedule it is a way to throw away the schedule.
PATIENCE = 0

EMA = True
EMA_DECAY = 0.9998
# Keep ResNet's original ImageNet stem. Adapting it for 64px inputs replaces
# the first convolution with a randomly initialised one, which discards
# pretrained features and makes every later layer run at 4x the spatial
# resolution for no accuracy gain at 224px.
ADAPT_STEM = False

MIRROR_EVERY = 10  # full checkpoint to Drive every N epochs; history every epoch
CALIBRATION_IMAGES = 200
EVAL_SAMPLES = 2000
BENCHMARK_BATCH_SIZES = (1, 8, 32)

DRIVE_FOLDER = "ML-Engineer-Challenge-results"

IN_COLAB = Path("/content").exists()
DATA_DIR = Path("/content/data") if IN_COLAB else Path.cwd() / "data"

MODEL_NAME = "resnet50-tiny-imagenet"
ONNX_NAME = f"{MODEL_NAME}.onnx"
INT8_NAME = f"{MODEL_NAME}_int8_static.onnx"
LABELS_NAME = "tiny_imagenet_labels.json"
BEST_CKPT = f"{ARCH}_tiny_imagenet_best.pt"
LAST_CKPT = f"{ARCH}_tiny_imagenet_last.pt"
HISTORY_NAME = f"{ARCH}_training_history.json"


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
class StageFailed(RuntimeError):
    """A required stage failed; everything after it is meaningless."""


_results: list[tuple[str, str, float, str]] = []


@contextmanager
def stage(title: str, *, required: bool = True):
    """Run a stage, timing it and recording the outcome.

    An optional stage that raises is recorded and skipped over. A required one
    aborts the run: continuing past a failed export only produces reports about
    a model that does not exist.
    """
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - started
        detail = f"{type(exc).__name__}: {exc}"
        _results.append((title, "FAILED" if required else "skipped", elapsed, detail))
        traceback.print_exc()
        if required:
            raise StageFailed(f"{title}: {detail}") from exc
        print(f"\n  SKIPPED ({detail})", flush=True)
    else:
        elapsed = time.perf_counter() - started
        _results.append((title, "ok", elapsed, ""))
        print(f"\n  done in {elapsed:.0f}s", flush=True)


def run(cmd: list[str], *, check: bool = True) -> int:
    """Stream a subprocess straight through, so training output is live."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    code = subprocess.call([str(c) for c in cmd])
    if check and code != 0:
        raise RuntimeError(f"command exited {code}: {' '.join(str(c) for c in cmd)}")
    return code


def copy_out(paths: list[Path], dest: Path) -> None:
    """Copy files to ``dest``, never raising - delivery is insurance, not the run."""
    if dest is None:
        return
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"  deliver failed ({dest}): {exc}", file=sys.stderr)
        return
    for source in paths:
        if not source.is_file():
            continue
        try:
            # Copy to a temporary name and rename: a copy interrupted partway
            # leaves a truncated file that looks valid, which is worse than
            # no copy at all.
            tmp = dest / (source.name + ".partial")
            shutil.copy2(source, tmp)
            tmp.replace(dest / source.name)
            print(f"  -> {dest.name}/{source.name}  ({source.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"  deliver failed ({source.name}): {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def show_gpu() -> None:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    print(out or "NO GPU DETECTED - everything below runs on CPU, far more slowly.")


def mount_drive(enabled: bool) -> Path | None:
    """Mount Drive and return the results folder, or abort.

    Deliberately the first thing that happens. Discovering that results cannot
    be saved is worth ten seconds at the start and is worth nothing at all
    after the training has finished.
    """
    if not enabled:
        print("Drive disabled (--no-drive). Results stay on this machine only.")
        return None

    if not IN_COLAB:
        local = Path.cwd() / "gpu_results"
        print(f"Not on Colab - delivering to {local} instead of Drive.")
        return local

    root = Path("/content/drive/MyDrive")
    if not root.exists():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as exc:
            raise RuntimeError(
                f"could not mount Drive: {exc}\n\n"
                "The mount needs an interactive popup, which a kernel driven from "
                "VS Code cannot show. Open this notebook in a browser Colab tab, or "
                "re-run with --no-drive and collect the results from the Files pane."
            ) from exc

    if not root.exists():
        raise RuntimeError("Drive mounted but /content/drive/MyDrive is not there.")

    dest = root / DRIVE_FOLDER
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Drive ready: My Drive / {DRIVE_FOLDER}")
    return dest


def get_repo() -> Path:
    """Clone, or force an existing checkout to match the remote.

    ``reset --hard`` rather than ``pull``: a shallow clone cannot always
    fast-forward, and the job here is "make this runtime match the remote",
    not "merge". Reusing a directory merely because it exists is how a runtime
    ends up training yesterday's configuration.
    """
    for candidate in (Path.cwd(), Path("/content") / REPO_NAME, Path.cwd() / REPO_NAME):
        if not (candidate / "models" / "training" / "train_classifier.py").is_file():
            continue
        print(f"existing checkout: {candidate}")
        subprocess.call(["git", "-C", str(candidate), "fetch", "--depth", "1", "origin", BRANCH])
        subprocess.call(["git", "-C", str(candidate), "reset", "--hard", f"origin/{BRANCH}"])
        return candidate.resolve()

    target = (Path("/content") if IN_COLAB else Path.cwd()) / REPO_NAME
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(target)])
    return target.resolve()


def install_dependencies() -> None:
    run([sys.executable, "scripts/install_for_gpu_runtime.py", "--with-tensorrt"])


def ensure_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "tiny-imagenet-200").exists():
        run(
            [
                sys.executable,
                "scripts/download_datasets.py",
                "--dataset",
                "tiny_imagenet",
                "--data-dir",
                str(DATA_DIR),
            ]
        )

    from models.training.dataset import TinyImageNetTrain, TinyImageNetVal, find_dataset_root

    root = find_dataset_root(DATA_DIR)
    train = TinyImageNetTrain(root)
    val = TinyImageNetVal(root, train.class_to_idx)
    distinct = len({label for _, label in val.samples})
    print(f"classes {len(train.classes)}  train {len(train):,}  val {len(val):,} ({distinct})")
    # Fail here rather than after an hour: the training script refuses a
    # partial dataset anyway, but this costs seconds.
    if not (len(train.classes) == 200 and len(train) == 100_000 and distinct == 200):
        raise RuntimeError("dataset is incomplete - delete it and re-run to download again")
    print("dataset verified COMPLETE")


def adopt_mirrored_checkpoint(artifacts: Path, mirror: Path | None) -> None:
    """Bring a mirrored checkpoint back, if it matches this configuration.

    Restarting a recycled runtime is the normal case, not the exception. What
    must never happen is adopting a checkpoint trained at another resolution
    or architecture: training would resume and converge to a number that
    looks legitimate and describes a different experiment.
    """
    if mirror is None:
        return
    local_last = artifacts / LAST_CKPT
    if local_last.exists():
        print("a local checkpoint is already present; leaving it alone")
        return

    candidates = [mirror / "checkpoints" / LAST_CKPT, mirror / "artifacts" / LAST_CKPT]
    source = next((c for c in candidates if c.is_file()), None)
    if source is None:
        print("no mirrored checkpoint found - training starts from epoch 1")
        return

    import torch

    state = torch.load(source, map_location="cpu", weights_only=False)
    config = state.get("config", {}) or {}
    mismatches = []
    if config.get("arch") not in (None, ARCH):
        mismatches.append(f"arch {config.get('arch')} != {ARCH}")
    if config.get("image_size") not in (None, IMAGE_SIZE):
        mismatches.append(f"image_size {config.get('image_size')} != {IMAGE_SIZE}")
    if config.get("adapt_stem") is not None and bool(config["adapt_stem"]) != ADAPT_STEM:
        mismatches.append(f"adapt_stem {config['adapt_stem']} != {ADAPT_STEM}")

    if mismatches:
        print(f"mirrored checkpoint at epoch {state.get('epoch')} is INCOMPATIBLE:")
        for line in mismatches:
            print(f"  - {line}")
        print("ignoring it; training starts from epoch 1")
        return

    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, local_last)
    for extra in (HISTORY_NAME, BEST_CKPT):
        found = next((c.parent / extra for c in candidates if (c.parent / extra).is_file()), None)
        if found:
            shutil.copy2(found, artifacts / extra)
    print(f"adopted mirrored checkpoint: epoch {state.get('epoch')}, resuming from there")


def train(artifacts: Path, mirror: Path | None) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "models.training.train_classifier",
        "--arch",
        ARCH,
        "--epochs",
        str(EPOCHS),
        "--image-size",
        str(IMAGE_SIZE),
        "--batch-size",
        str(BATCH_SIZE),
        "--lr",
        LEARNING_RATE,
        "--num-workers",
        str(NUM_WORKERS),
        "--scheduler",
        "cosine",
        "--warmup-ratio",
        "0.05",
        "--grad-clip",
        "1.0",
        "--label-smoothing",
        "0.1",
        "--patience",
        str(PATIENCE),
        "--data-dir",
        str(DATA_DIR),
        "--device",
        "cuda",
    ]
    if EMA:
        cmd += ["--ema", "--ema-decay", str(EMA_DECAY)]
    if not ADAPT_STEM:
        cmd += ["--no-stem-adapt"]
    if mirror is not None:
        cmd += ["--mirror-dir", str(mirror / "checkpoints"), "--mirror-every", str(MIRROR_EVERY)]
    run(cmd)

    history = artifacts / HISTORY_NAME
    if history.is_file():
        data = json.loads(history.read_text(encoding="utf-8"))
        print(
            f"\nBEST top-1 {data.get('best_top1')}% at epoch {data.get('best_epoch')} "
            f"of {len(data.get('epochs', []))}, stopped_early={data.get('stopped_early')}"
        )


def export_onnx(artifacts: Path) -> None:
    import torch

    from models.optimization.export_onnx import export_to_onnx
    from models.training.train_classifier import adapt_stem_for_small_images, build_model

    ckpt = torch.load(artifacts / BEST_CKPT, map_location="cpu", weights_only=False)
    print(
        f"checkpoint epoch {ckpt['epoch']}, top-1 {ckpt['val_top1']:.2f}%, "
        f"weights={ckpt.get('weights', 'raw')}"
    )

    model = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False)
    if ckpt.get("stem_adapted"):
        adapt_stem_for_small_images(model)
    model.load_state_dict(ckpt["model_state_dict"])

    result = export_to_onnx(
        model,
        artifacts / ONNX_NAME,
        input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        name=MODEL_NAME,
    )
    print(
        f"onnx {result.size_mb:.1f} MB  max diff {result.max_abs_diff:.2e}  "
        f"verified={result.verified}"
    )
    for note in result.notes:
        print(f"  note: {note}")
    if not result.verified:
        raise RuntimeError("ONNX export does not match PyTorch within tolerance")

    (artifacts / LABELS_NAME).write_text(json.dumps(ckpt["class_names"]), encoding="utf-8")
    print(f"labels: {len(ckpt['class_names'])} classes")


def quantize(artifacts: Path) -> None:
    from api.utils.image_processing import PreprocessConfig
    from models.optimization.quantize import quantize_onnx_static

    calibration = DATA_DIR / "tiny-imagenet-200" / "tiny-imagenet-200" / "val" / "images"
    result = quantize_onnx_static(
        artifacts / ONNX_NAME,
        calibration,
        PreprocessConfig(size=(IMAGE_SIZE, IMAGE_SIZE)),
        num_calibration=CALIBRATION_IMAGES,
    )
    print(result.summary())
    for note in result.notes:
        print(f"  note: {note}")


def build_tensorrt(artifacts: Path, reports: Path) -> None:
    from models.optimization.export_tensorrt import (
        UnsupportedPrecisionError,
        benchmark_engine,
        build_engine,
        tensorrt_available,
    )

    available, reason = tensorrt_available()
    print(f"TensorRT available: {available} ({reason})")
    if not available:
        raise RuntimeError(reason)

    summary = {}
    for precision in ("fp16", "fp32"):
        print(f"\n--- {precision} ---")
        try:
            res = build_engine(
                artifacts / ONNX_NAME,
                precision=precision,
                max_batch_size=32,
                workspace_gb=8.0,
            )
            print(res.summary())
            bench = benchmark_engine(
                Path(res.engine_path),
                input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                iterations=200,
                warmup=50,
            )
            print(
                f"  p50 {bench['p50_ms']:.3f} ms  p95 {bench['p95_ms']:.3f} ms  "
                f"{bench['throughput_ips']:.0f} img/s"
            )
            summary[precision] = {"build": res.to_dict(), "benchmark": bench}
        except UnsupportedPrecisionError as exc:
            # Not a failure: this TensorRT build cannot express the precision
            # from this graph. Recorded as a skip so the run reads honestly.
            print(f"  SKIPPED: {exc}")
            summary[precision] = {"skipped": str(exc)}
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            summary[precision] = {"failed": f"{type(exc).__name__}: {exc}"}

    (reports / "tensorrt.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")


def benchmark(artifacts: Path, reports: Path) -> None:
    from models.optimization.benchmark import benchmark_onnx, environment_info, render_markdown

    results = []
    for name in (ONNX_NAME, INT8_NAME):
        path = artifacts / name
        if not path.exists():
            continue
        print(f"benchmarking {name} ...")
        results.extend(
            benchmark_onnx(
                path,
                input_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
                batch_sizes=BENCHMARK_BATCH_SIZES,
                iterations=100,
                warmup=20,
                device="cuda",
            )
        )

    for r in sorted(results, key=lambda x: (x.name, x.batch_size)):
        print(r.summary())

    # Name the report after the device actually used. ONNX Runtime silently
    # falls back to CPU when the CUDA provider is missing, and a file called
    # BENCHMARKS_GPU.md full of CPU numbers outlives the session that made it.
    devices = {r.device for r in results}
    if devices == {"cuda"}:
        out = reports / "BENCHMARKS_GPU.md"
    else:
        out = reports / "BENCHMARKS_GPU_CPU_FALLBACK.md"
        print(f"\nWARNING: ran on {', '.join(sorted(devices))}, not cuda. NOT GPU numbers.")

    out.write_text(render_markdown(results, environment_info()), encoding="utf-8")
    print(f"wrote {out}")


def register_and_validate(reports: Path) -> None:
    from models.registry import Registry
    from models.validation.validate import load_eval_samples, validate_model

    Registry().register(
        name=MODEL_NAME,
        version="1.0.0",
        task="classification",
        artifacts={"onnx": ONNX_NAME, "onnx_int8": INT8_NAME},
        preprocess="tiny_imagenet",
        labels_file=LABELS_NAME,
        num_classes=200,
        input_shape=[1, 3, IMAGE_SIZE, IMAGE_SIZE],
        description=(
            "ResNet-50 fine-tuned on Tiny-ImageNet (200 classes) with mixed "
            "precision, gradient clipping and cosine LR scheduling."
        ),
        overwrite=True,
    )

    samples = load_eval_samples(DATA_DIR, limit=EVAL_SAMPLES)
    report = validate_model(f"{MODEL_NAME}:1.0.0", eval_samples=samples, max_p95_ms=1000)
    print(report.summary())
    for check in report.checks:
        mark = (
            " ok " if check["passed"] else ("FAIL" if check["severity"] == "critical" else "warn")
        )
        print(f"  [{mark}] {check['name']:<20} {check['message']}")
    print("\nmetrics:")
    for key, value in sorted(report.metrics.items()):
        print(f"  {key:<28} {value}")

    (reports / "validation.json").write_text(
        json.dumps([report.to_dict()], indent=2), encoding="utf-8"
    )


def deliver(repo: Path, mirror: Path | None) -> None:
    """Copy artifacts and reports to Drive, and leave a zip beside them."""
    if mirror is None:
        print("no delivery target (--no-drive)")
        return

    artifacts = repo / "models" / "artifacts"
    reports = repo / "benchmarks" / "reports"

    wanted = [artifacts / n for n in (BEST_CKPT, HISTORY_NAME, ONNX_NAME, INT8_NAME, LABELS_NAME)]
    # Engine binaries are built for one GPU architecture and TensorRT version
    # and will not load anywhere else, so only their metadata travels.
    wanted += sorted(artifacts.glob("*.engine.json"))
    copy_out(wanted, mirror / "artifacts")
    copy_out([p for p in reports.glob("*") if p.is_file()], mirror / "reports")
    copy_out([repo / "models" / "registry.json"], mirror)

    try:
        # /content, not /tmp: it shows up in Colab's Files pane, which is the
        # fallback route for getting results out when Drive cannot be mounted.
        staging = Path("/content/gpu_results") if IN_COLAB else repo / ".gpu_results"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "artifacts").mkdir(parents=True)
        (staging / "reports").mkdir(parents=True)
        for path in wanted:
            if path.is_file():
                shutil.copy2(path, staging / "artifacts" / path.name)
        for path in reports.glob("*"):
            if path.is_file():
                shutil.copy2(path, staging / "reports" / path.name)
        shutil.copy2(repo / "models" / "registry.json", staging / "registry.json")
        archive = shutil.make_archive(str(staging), "zip", staging)
        shutil.copy2(archive, mirror / Path(archive).name)
        print(f"  -> {Path(archive).name}  ({Path(archive).stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        # The individual files are already delivered; the zip is a convenience.
        print(f"  zip failed ({exc}) - individual files are on Drive regardless")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-drive", action="store_true", help="Do not mount or write to Drive.")
    parser.add_argument("--skip-install", action="store_true", help="Dependencies already present.")
    parser.add_argument("--skip-train", action="store_true", help="Reuse the existing checkpoint.")
    parser.add_argument("--skip-tensorrt", action="store_true")
    args = parser.parse_args()

    overall = time.perf_counter()
    print(
        f"config: {ARCH} {IMAGE_SIZE}px  {EPOCHS} epochs  batch {BATCH_SIZE}  lr "
        f"{LEARNING_RATE}  ema={EMA}  patience={PATIENCE}"
    )

    with stage("1. GPU", required=False):
        show_gpu()

    mirror: Path | None = None
    with stage("2. Mount Drive"):
        mirror = mount_drive(not args.no_drive)

    with stage("3. Repository"):
        repo = get_repo()
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        print(f"working directory: {repo}")

    artifacts = repo / "models" / "artifacts"
    reports = repo / "benchmarks" / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        with stage("4. Dependencies"):
            install_dependencies()

    with stage("5. Dataset"):
        ensure_dataset()

    if not args.skip_train:
        with stage("6. Recover a mirrored checkpoint", required=False):
            adopt_mirrored_checkpoint(artifacts, mirror)

        with stage(f"7. Train ({EPOCHS} epochs at {IMAGE_SIZE}px)"):
            train(artifacts, mirror)
        copy_out(
            [artifacts / BEST_CKPT, artifacts / HISTORY_NAME],
            mirror / "artifacts" if mirror else None,
        )

    with stage("8. Export to ONNX"):
        export_onnx(artifacts)
    copy_out(
        [artifacts / ONNX_NAME, artifacts / LABELS_NAME], mirror / "artifacts" if mirror else None
    )

    with stage("9. Quantize to INT8", required=False):
        quantize(artifacts)
    copy_out([artifacts / INT8_NAME], mirror / "artifacts" if mirror else None)

    if not args.skip_tensorrt:
        with stage("10. TensorRT engines", required=False):
            build_tensorrt(artifacts, reports)

    with stage("11. Benchmark", required=False):
        benchmark(artifacts, reports)

    with stage("12. Register and validate", required=False):
        register_and_validate(reports)

    with stage("13. Deliver everything", required=False):
        deliver(repo, mirror)

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for title, status, elapsed, detail in _results:
        line = f"  [{status:>7}] {title:<42} {elapsed:6.0f}s"
        print(line + (f"  {detail}" if detail else ""))
    print(f"\n  total {time.perf_counter() - overall:.0f}s")

    history = artifacts / HISTORY_NAME
    if history.is_file():
        data = json.loads(history.read_text(encoding="utf-8"))
        epochs = data.get("epochs", [])
        best = max(epochs, key=lambda e: e["val_top1"]) if epochs else {}
        print(
            f"\n  BEST top-1 {data.get('best_top1')}%  top-5 {best.get('val_top5')}%  "
            f"at epoch {data.get('best_epoch')} of {len(epochs)}"
        )
        print(f"  early stop : {data.get('stopped_early')}")
    if mirror is not None:
        print(f"\n  results: {mirror}")

    failed = [t for t, s, _, _ in _results if s == "FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageFailed as exc:
        print(f"\nABORTED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
