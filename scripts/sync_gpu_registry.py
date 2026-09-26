"""Write the GPU overlay's model registry from models/registry.json.

The GPU overlay builds a TensorRT engine at pod start, but the API only loads
an engine that the registry lists as a ``tensorrt`` artifact. The shared
registry cannot list it: CPU deployments have no engine. So the overlay ships
its own copy, identical except for that one artifact, and kustomize cannot
read files outside the overlay directory, so the copy has to be a file.

The engine's file name is read from the overlay's build step, so the two
cannot disagree.

    python scripts/sync_gpu_registry.py           # write the copy
    python scripts/sync_gpu_registry.py --check   # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_REGISTRY = REPO_ROOT / "models" / "registry.json"
OVERLAY = REPO_ROOT / "k8s" / "overlays" / "gpu"
GPU_REGISTRY = OVERLAY / "registry.json"
MODEL = "resnet50-tiny-imagenet"


def engine_file() -> str:
    """The engine the overlay's build-engine init container writes."""
    patch = yaml.safe_load((OVERLAY / "gpu-patch.yaml").read_text(encoding="utf-8"))
    init = patch["spec"]["template"]["spec"]["initContainers"]
    command = next(c for c in init if c["name"] == "build-engine")["command"]
    return Path(command[command.index("--output") + 1]).name


def build() -> dict[str, Any]:
    registry = copy.deepcopy(json.loads(BASE_REGISTRY.read_text(encoding="utf-8")))
    entries = [m for m in registry["models"] if m["name"] == MODEL]
    if not entries:
        raise SystemExit(f"{MODEL} is not in {BASE_REGISTRY}")
    for entry in entries:
        entry["artifacts"]["tensorrt"] = engine_file()
    return registry


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="Fail if the copy is stale.")
    args = parser.parse_args()

    expected = render()
    if args.check:
        current = GPU_REGISTRY.read_text(encoding="utf-8") if GPU_REGISTRY.exists() else ""
        if current != expected:
            print(f"STALE: {GPU_REGISTRY} (run python scripts/sync_gpu_registry.py)")
            return 1
        print(f"up to date: {GPU_REGISTRY}")
        return 0
    GPU_REGISTRY.write_text(expected, encoding="utf-8")
    print(f"wrote {GPU_REGISTRY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
