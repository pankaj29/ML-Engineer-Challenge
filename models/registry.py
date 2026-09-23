"""Model registry management.

Plain English:
    The registry is a single JSON file listing every model this system can
    serve: its name, version, task, where its artifact files live, how to
    preprocess input for it, how accurate it measured, and what it is bad at.

    It is the contract between *training* (which produces models) and
    *serving* (which runs them). Training writes entries here; the API reads
    them. Neither needs to know anything else about the other.

Why a JSON file rather than a database table? Because the registry must be
readable before the database connection exists — the API loads models during
startup, and a model registry that depends on a healthy database gives you a
service that cannot start when the database is slow. The file is small,
versionable in git, and diffable in a pull request. Registrations are *also*
recorded in PostgreSQL (:class:`db.models.ModelVersionRecord`) for audit
history, but that path is never on the startup critical path.

Usage::

    python -m models.registry list
    python -m models.registry register --name resnet18 --version 1.0.0 \
        --task classification --onnx resnet18.onnx --labels imagenet_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REGISTRY = REPO_ROOT / "models" / "registry.json"
DEFAULT_ARTIFACTS = REPO_ROOT / "models" / "artifacts"


class Registry:
    """Read/write access to the model registry file."""

    def __init__(self, path: Path = DEFAULT_REGISTRY) -> None:
        self.path = Path(path)
        self.models: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        """Read the registry from disk, tolerating a missing file."""
        if not self.path.exists():
            self.models = []
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.models = raw.get("models", []) if isinstance(raw, dict) else list(raw)

    def save(self) -> None:
        """Write the registry back to disk.

        Written atomically via a temporary file and a rename. A process killed
        halfway through a plain write leaves a truncated JSON file, which
        would stop the API from loading *any* model on its next start.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "models": self.models,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def find(self, name: str, version: str) -> dict[str, Any] | None:
        """Return the entry for ``name:version``, if present."""
        for entry in self.models:
            if entry.get("name") == name and entry.get("version") == version:
                return entry
        return None

    def register(
        self,
        *,
        name: str,
        version: str,
        task: str,
        artifacts: dict[str, str],
        preprocess: str = "imagenet_224",
        labels_file: str | None = None,
        num_classes: int | None = None,
        input_shape: list[int] | None = None,
        metrics: dict[str, float] | None = None,
        limitations: list[str] | None = None,
        description: str | None = None,
        is_default: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Add or update a model version.

        Args:
            overwrite: Replace an existing entry with the same name/version.
                Without this, re-registering raises — because silently
                replacing a version is how you end up unable to explain which
                weights produced last week's predictions.

        Raises:
            ValueError: The entry already exists and ``overwrite`` is False.
        """
        existing = self.find(name, version)
        if existing and not overwrite:
            raise ValueError(
                f"{name}:{version} is already registered. Pass overwrite=True to replace it, "
                "or register a new version instead."
            )

        entry = {
            "name": name,
            "version": version,
            "task": task,
            "artifacts": artifacts,
            "preprocess": preprocess,
            "labels_file": labels_file,
            "num_classes": num_classes,
            "input_shape": input_shape,
            "metrics": metrics or {},
            "limitations": limitations or [],
            "description": description,
            "is_default": is_default,
            "registered_at": datetime.now(UTC).isoformat(),
            "status": "active",
        }

        if is_default:
            # Only one default per task, so clear the flag on siblings.
            for other in self.models:
                if other.get("task") == task and other is not existing:
                    other["is_default"] = False

        if existing:
            self.models[self.models.index(existing)] = entry
        else:
            self.models.append(entry)

        self.save()
        return entry

    def retire(self, name: str, version: str) -> bool:
        """Mark a version inactive so the API stops serving it.

        The entry is kept rather than deleted, preserving the record of what
        was once live.
        """
        entry = self.find(name, version)
        if entry is None:
            return False
        entry["status"] = "retired"
        entry["retired_at"] = datetime.now(UTC).isoformat()
        self.save()
        return True

    def validate(self, artifacts_dir: Path = DEFAULT_ARTIFACTS) -> list[str]:
        """Check that every registered artifact file actually exists.

        Catches the single most common deployment failure: a registry entry
        pointing at a file that was never copied into the image.
        """
        problems: list[str] = []
        for entry in self.models:
            if entry.get("status") != "active":
                continue
            key = f"{entry['name']}:{entry['version']}"
            if not entry.get("artifacts"):
                problems.append(f"{key}: no artifacts registered")
                continue
            for fmt, rel in entry["artifacts"].items():
                path = Path(rel)
                if not path.is_absolute():
                    path = artifacts_dir / path
                if not path.exists():
                    problems.append(f"{key}: {fmt} artifact missing at {path}")
            if entry.get("labels_file"):
                labels = Path(entry["labels_file"])
                if not labels.is_absolute():
                    labels = artifacts_dir / labels
                if not labels.exists():
                    problems.append(f"{key}: labels file missing at {labels}")
        return problems


def _cmd_list(args: argparse.Namespace) -> int:
    registry = Registry(args.registry)
    if not registry.models:
        print("registry is empty")
        return 0

    print(f"{'NAME':<30} {'VERSION':<10} {'TASK':<16} {'DEFAULT':<8} {'FORMATS'}")
    print("-" * 90)
    for entry in registry.models:
        if entry.get("status") != "active" and not args.all:
            continue
        print(
            f"{entry['name']:<30} {entry['version']:<10} {entry['task']:<16} "
            f"{'yes' if entry.get('is_default') else '':<8} "
            f"{','.join(sorted(entry.get('artifacts', {})))}"
        )
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    registry = Registry(args.registry)
    artifacts: dict[str, str] = {}
    for fmt in ("torch", "torch_int8", "onnx", "onnx_int8", "tensorrt"):
        value = getattr(args, fmt, None)
        if value:
            artifacts[fmt] = value

    if not artifacts:
        print("error: register at least one artifact (--onnx, --torch, ...)", file=sys.stderr)
        return 2

    entry = registry.register(
        name=args.name,
        version=args.version,
        task=args.task,
        artifacts=artifacts,
        preprocess=args.preprocess,
        labels_file=args.labels,
        num_classes=args.num_classes,
        input_shape=[int(x) for x in args.input_shape.split(",")] if args.input_shape else None,
        description=args.description,
        is_default=args.default,
        overwrite=args.overwrite,
    )
    print(f"registered {entry['name']}:{entry['version']} ({entry['task']})")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    registry = Registry(args.registry)
    problems = registry.validate(args.artifacts_dir)
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"all {len([m for m in registry.models if m.get('status') == 'active'])} active models validate"
    )
    return 0


def _cmd_retire(args: argparse.Namespace) -> int:
    registry = Registry(args.registry)
    if registry.retire(args.name, args.version):
        print(f"retired {args.name}:{args.version}")
        return 0
    print(f"error: {args.name}:{args.version} is not registered", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the model registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registered models.")
    p_list.add_argument("--all", action="store_true", help="Include retired versions.")
    p_list.set_defaults(func=_cmd_list)

    p_reg = sub.add_parser("register", help="Register a model version.")
    p_reg.add_argument("--name", required=True)
    p_reg.add_argument("--version", required=True)
    p_reg.add_argument(
        "--task", required=True, choices=["classification", "detection", "similarity"]
    )
    p_reg.add_argument("--onnx", help="Path to the ONNX artifact, relative to models/artifacts.")
    p_reg.add_argument("--onnx-int8", dest="onnx_int8")
    p_reg.add_argument("--torch")
    p_reg.add_argument("--torch-int8", dest="torch_int8")
    p_reg.add_argument("--tensorrt")
    p_reg.add_argument("--labels", help="Labels JSON file, relative to models/artifacts.")
    p_reg.add_argument("--preprocess", default="imagenet_224")
    p_reg.add_argument("--num-classes", type=int)
    p_reg.add_argument("--input-shape", help="Comma-separated, e.g. 1,3,224,224")
    p_reg.add_argument("--description")
    p_reg.add_argument("--default", action="store_true", help="Make this the task default.")
    p_reg.add_argument("--overwrite", action="store_true")
    p_reg.set_defaults(func=_cmd_register)

    p_val = sub.add_parser("validate", help="Check that all artifact files exist.")
    p_val.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    p_val.set_defaults(func=_cmd_validate)

    p_ret = sub.add_parser("retire", help="Retire a model version.")
    p_ret.add_argument("--name", required=True)
    p_ret.add_argument("--version", required=True)
    p_ret.set_defaults(func=_cmd_retire)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
