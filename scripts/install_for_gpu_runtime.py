"""Install this project's dependencies on a hosted GPU runtime.

Plain English:
    Colab (and any similar hosted GPU) arrives with PyTorch, CUDA and cuDNN
    already installed, built against each other and against the specific
    driver on that machine. This installs everything the project needs
    *around* that, without disturbing it.

WHY NOT JUST ``pip install -r requirements-train.txt``?

    Because ``requirements.txt`` pins ``torch~=2.9.0``, and a Colab A100
    runtime ships torch 2.11 built for CUDA 12.8. Installing the pin would
    **downgrade torch**, and pip would almost certainly resolve to the default
    PyPI wheel — which is CPU-only. The result is a working A100 session that
    silently loses its GPU, with no error message, discovered several minutes
    into a training run when it is inexplicably slow.

    So the requirements files stay the single source of truth for *what* the
    project needs, and this script skips the handful of packages the runtime
    already provides correctly.

WHAT IS SKIPPED, AND WHY

    torch, torchvision   The runtime's build is matched to its CUDA and
                         driver. Ours is not.
    onnxruntime          The CPU package. ``onnxruntime-gpu`` replaces it, and
                         installing both makes which one wins ambiguous.

PINS ARE RELAXED TO MINIMUMS

    The requirements files use ``~=`` ("compatible release"), which is right
    for a Docker image we control: ``timm~=1.0.11`` means ``>=1.0.11,<1.1.0``.

    On a hosted runtime it is wrong. If the runtime ships something newer,
    that pin forces a *downgrade* - and if no wheel exists for the runtime's
    Python version, pip falls back to building from source, which needs a
    toolchain the runtime may not have.

    ``--only-binary`` covers the related trap: --prefer-binary only prefers a
    wheel for a version pip has already chosen, so pip can still SELECT a
    version that has no wheel at all and then try to build it.

    So every ``~=`` becomes ``>=`` here. That still guarantees "at least the
    version the tests ran against", while letting the runtime keep whatever
    newer build it already has. Pass ``--strict-pins`` to install the exact
    ranges instead.

Usage::

    python scripts/install_for_gpu_runtime.py            # install
    python scripts/install_for_gpu_runtime.py --dry-run  # show the plan only
    python scripts/install_for_gpu_runtime.py --with-tensorrt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages the hosted runtime already provides, correctly built. Compared
# against the requirement's bare name, lower-cased.
RUNTIME_PROVIDES = {
    "torch": "the runtime's build is matched to its CUDA version and driver",
    "torchvision": "must match the runtime's torch build",
    "onnxruntime": "superseded by onnxruntime-gpu, which is installed instead",
    # numpy shares a C ABI with torch. The runtime's numpy was chosen to match
    # its torch build; pinning ours risks the same silent breakage as pinning
    # torch itself. The verification step at the end catches it if this
    # assumption is ever wrong.
    "numpy": "ABI-coupled to the runtime's torch build",
}

# Packages that must come from a wheel or not at all. Each ships a native build
# in its sdist that cannot succeed on a stock hosted runtime.
#
# pycuda is deliberately NOT here: it legitimately builds from source, and the
# runtime has the toolchain it needs.
BINARY_ONLY = {"onnx", "onnxruntime-gpu", "protobuf", "onnxscript"}


# Serving-only dependencies. Harmless to install, but they are pure overhead
# on a machine whose only job is to train and export, so they are skipped
# unless --all is passed.
SERVING_ONLY = {
    "fastapi",
    "uvicorn",
    "python-multipart",
    "redis",
    "celery",
    "sqlalchemy",
    "asyncpg",
    "psycopg",
    "alembic",
    "passlib",
    "pyjwt",
}


def requirement_name(line: str) -> str:
    """Extract the bare package name from a requirement line.

    ``redis[hiredis]~=5.2.0  # comment`` -> ``redis``
    """
    line = line.split("#", 1)[0].strip()
    if not line:
        return ""
    # Strip extras, then any version specifier.
    name = re.split(r"[\[<>=!~;]", line, maxsplit=1)[0]
    return name.strip().lower()


def relax_pin(line: str) -> str:
    """Turn a compatible-release pin into a minimum-version one.

    ``onnx~=1.17.0`` -> ``onnx>=1.17.0``

    Why: ``~=1.17.0`` caps at ``<1.18.0``, so pip must DOWNGRADE a runtime
    that already ships something newer. When no wheel exists for that older
    version on the runtime's Python, pip builds from source and fails. A
    minimum keeps the guarantee that matters (not older than what we tested)
    without fighting the base image.
    """
    return line.replace("~=", ">=", 1) if "~=" in line else line


def read_requirements(path: Path, _seen: set[Path] | None = None) -> list[str]:
    """Read a requirements file, following ``-r`` includes.

    Returns the requirement lines with comments and blanks removed, in order,
    de-duplicated by package name (first occurrence wins).
    """
    _seen = _seen if _seen is not None else set()
    path = path.resolve()
    if path in _seen or not path.is_file():
        return []
    _seen.add(path)

    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith(("-r ", "--requirement ")):
            included = stripped.split(maxsplit=1)[1].strip()
            lines.extend(read_requirements(path.parent / included, _seen))
        elif stripped.startswith("-"):
            continue  # other pip flags are not our business
        else:
            lines.append(stripped)

    # De-duplicate, keeping the first pin seen for each package.
    out: list[str] = []
    taken: set[str] = set()
    for line in lines:
        name = requirement_name(line)
        if name and name not in taken:
            taken.add(name)
            out.append(line)
    return out


def build_plan(
    *,
    include_serving: bool = False,
    with_tensorrt: bool = False,
    strict_pins: bool = False,
) -> dict:
    """Decide what to install, what to skip, and why."""
    requested = read_requirements(REPO_ROOT / "requirements-train.txt")
    if not strict_pins:
        requested = [relax_pin(line) for line in requested]

    install: list[str] = []
    skipped: list[tuple[str, str]] = []

    for line in requested:
        name = requirement_name(line)
        if name in RUNTIME_PROVIDES:
            skipped.append((line, RUNTIME_PROVIDES[name]))
        elif name in SERVING_ONLY and not include_serving:
            skipped.append((line, "serving-only; not needed to train or export"))
        else:
            install.append(line)

    # The GPU extras. onnxruntime-gpu always; TensorRT on request, because it
    # is a large download and only needed for the engine-export step.
    gpu_extras = [
        line if strict_pins else relax_pin(line)
        for line in read_requirements(REPO_ROOT / "requirements-gpu.txt")
        if requirement_name(line) in ({"onnxruntime-gpu", "tensorrt", "pycuda"})
    ]
    for line in gpu_extras:
        name = requirement_name(line)
        if name in {"tensorrt", "pycuda"} and not with_tensorrt:
            skipped.append((line, "TensorRT extras; pass --with-tensorrt to include"))
        else:
            install.append(line)

    return {"install": install, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, install nothing.")
    parser.add_argument(
        "--with-tensorrt", action="store_true", help="Also install tensorrt and pycuda."
    )
    parser.add_argument(
        "--all",
        dest="include_serving",
        action="store_true",
        help="Include the serving-only dependencies (FastAPI, Redis, Celery, ...).",
    )
    parser.add_argument(
        "--strict-pins",
        action="store_true",
        help=(
            "Install the exact ~= ranges from the requirements files. Risks "
            "downgrading packages the runtime ships, which can trigger a "
            "source build that fails."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Less pip output.")
    args = parser.parse_args()

    plan = build_plan(
        include_serving=args.include_serving,
        with_tensorrt=args.with_tensorrt,
        strict_pins=args.strict_pins,
    )

    print("Source of truth  : requirements-train.txt (+ requirements-gpu.txt)")
    print(
        "Pin strategy     : "
        + (
            "exact (~=) - may downgrade runtime packages"
            if args.strict_pins
            else "minimum (>=) - keeps newer versions the runtime already has"
        )
    )
    print(f"Will install     : {len(plan['install'])} package(s)")
    for line in plan["install"]:
        print(f"    + {line}")

    print(f"\nWill skip        : {len(plan['skipped'])} package(s)")
    for line, reason in plan["skipped"]:
        print(f"    - {requirement_name(line):<18} {reason}")

    if args.dry_run:
        print("\n(dry run - nothing installed)")
        return 0

    # --- Stage 1: the packages that must never be built from source --------
    #
    # onnx ships a CMake/protobuf build in its sdist. On a hosted runtime that
    # build fails slowly, with the real compiler error buried thousands of
    # lines up. Installing these in their OWN pip call with --only-binary=:all:
    # makes that outcome impossible: if no wheel matches this runtime's Python,
    # pip says "no matching distribution" immediately instead of starting a
    # build it cannot finish.
    #
    # A single combined call with a per-package --only-binary list was tried
    # first and did not hold: once the resolver started backtracking across the
    # other 19 requirements it still selected an sdist. Isolating these removes
    # the resolver pressure as well as the ambiguity.
    #
    # This stage is NEVER quiet. Its failure is the one that has cost the most
    # time, and -q hides the line that explains it.
    binary_only = [line for line in plan["install"] if requirement_name(line) in BINARY_ONLY]
    rest = [line for line in plan["install"] if requirement_name(line) not in BINARY_ONLY]

    if binary_only:
        cmd = [sys.executable, "-m", "pip", "install", "--only-binary=:all:", *binary_only]
        print(f"\nStage 1/2: wheels-only ({len(binary_only)} packages) ...")
        print("    $ " + " ".join(cmd[1:]))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                "\nStage 1 FAILED. This stage forbids source builds, so the error\n"
                "above is the real one - no CMake output is hiding it.\n"
                "\nIf it says 'no matching distribution', this runtime's Python has\n"
                "no wheel for the required version. Find out which, with:\n"
                "    python -VV\n"
                "    pip index versions onnx",
                file=sys.stderr,
            )
            return result.returncode

    # --- Stage 2: everything else ------------------------------------------
    cmd = [sys.executable, "-m", "pip", "install", "--prefer-binary"]
    if args.quiet:
        cmd.append("-q")
    cmd.extend(rest)

    print(f"\nStage 2/2: pip install ({len(rest)} packages) ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\npip install FAILED", file=sys.stderr)
        return result.returncode

    if restart_required():
        return 2

    return verify()


def restart_required() -> bool:
    """True if a C extension was replaced underneath a live interpreter.

    protobuf is the one that matters. It is a compiled extension, so the copy
    already imported into this process stays live no matter what pip writes to
    disk. Every later onnx call then uses the OLD protobuf and fails, while the
    installation looks perfect.

    Without this check that surfaces as a baffling ONNX round-trip error with a
    correct-looking `pip list`. Naming it directly is worth the twenty lines.
    """
    import importlib.metadata as md

    try:
        on_disk = md.version("protobuf")
    except md.PackageNotFoundError:
        return False

    loaded = sys.modules.get("google.protobuf")
    in_process = getattr(loaded, "__version__", None) if loaded else None

    if in_process is None or in_process == on_disk:
        return False

    print(
        f"\n{'=' * 70}\n"
        f"RESTART THE RUNTIME BEFORE CONTINUING\n"
        f"{'=' * 70}\n"
        f"protobuf on disk    : {on_disk}\n"
        f"protobuf in memory  : {in_process}\n\n"
        "protobuf is a compiled extension. The version already imported into\n"
        "this process cannot be swapped out, so ONNX will keep using the old\n"
        "one and fail, even though the install succeeded.\n\n"
        "In Colab: Runtime > Restart session. Then re-run the 'Get the code'\n"
        "cell and continue. This script is safe to re-run; it will find\n"
        "everything already installed and skip straight to verification.\n"
        f"{'=' * 70}",
        file=sys.stderr,
    )
    return True


def verify() -> int:
    """Prove the installed stack actually works, rather than assuming it.

    "pip exited 0" is not evidence. Two real failures on a hosted runtime were
    invisible to it:

    * Installing our torch pin silently replaced the CUDA build with a
      CPU-only wheel. pip was perfectly happy; the GPU was simply gone.
    * A protobuf version that satisfied pip's resolver still broke onnx,
      because onnx needs >= 6.31.1 and the resolver had no way to know that
      mattered more than host packages asking for less.

    So this runs the operations the pipeline actually depends on.
    """
    print("\n--- verification ---")

    # 1. torch, and whether it can still see the GPU.
    try:
        import torch
    except ImportError as exc:
        print(f"torch            : NOT IMPORTABLE ({exc})", file=sys.stderr)
        return 1

    print(f"torch            : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    else:
        print(
            "\nWARNING: torch can no longer see the GPU. Something in this "
            "install replaced the runtime's CUDA build.\n"
            "Restart the runtime and re-run this script.",
            file=sys.stderr,
        )
        return 1

    # 2. onnx and protobuf, whose versions must agree with each other.
    try:
        import google.protobuf as protobuf_pkg
        import onnx
    except ImportError as exc:
        print(f"onnx/protobuf    : NOT IMPORTABLE ({exc})", file=sys.stderr)
        return 1

    print(f"onnx             : {onnx.__version__}")
    print(f"protobuf         : {protobuf_pkg.__version__}")

    # 3. The real test: export -> check -> load -> infer. This is the path
    #    protobuf is used for, and the only way to tell "genuinely compatible"
    #    from "merely co-installed".
    try:
        import tempfile

        import numpy as np
        import onnxruntime as ort

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 8 * 8, 4),
        ).eval()

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "probe.onnx")
            kwargs = {}
            if "dynamo" in torch.onnx.export.__code__.co_varnames:
                kwargs["dynamo"] = False
            torch.onnx.export(model, torch.randn(1, 3, 8, 8), path, **kwargs)
            onnx.checker.check_model(onnx.load(path))

            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            session.run(
                None,
                {session.get_inputs()[0].name: np.random.randn(1, 3, 8, 8).astype("float32")},
            )

        print("ONNX round-trip  : OK (export, check, load, infer)")
        providers = ort.get_available_providers()
        print(f"ORT providers    : {providers}")
        if "CUDAExecutionProvider" not in providers:
            print("  note: ONNX Runtime has no CUDA provider; its benchmarks will use CPU")
    except Exception as exc:
        print(f"\nONNX round-trip  : FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nThis almost always means onnx and protobuf disagree. onnx >= 1.23\n"
            "needs protobuf >= 6.31.1; some host packages ask for less. Force the\n"
            "declared range, then RESTART the runtime (protobuf is a C extension\n"
            "and will not reload in place):\n"
            '    pip install --prefer-binary "protobuf>=6.31.1,<7"',
            file=sys.stderr,
        )
        return 1

    print("\nReady.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
