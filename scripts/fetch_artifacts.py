"""Fetch model artifacts into a pod at startup.

Plain English:
    Kubernetes pods start with an empty disk. The weights have to come from
    somewhere, and this is the somewhere: an init container runs this, it
    downloads what the registry asks for into a volume the app container
    shares, and only then does the API start.

Why not bake them into the image. The artifacts are 388 MB across eight files.
Putting them in the image ties the model version to the image version, so
shipping new weights means rebuilding and redeploying the service, and every
rollback of a code change also rolls back the model. Keeping them separate
means the image is small and the model is versioned on its own.

Why not a shared volume. A ReadOnlyMany PVC needs a filesystem volume that
most clusters do not have by default, and it becomes a single point of failure
that every replica mounts. Downloading per pod costs disk and a few seconds of
startup, and removes that dependency entirely.

**Checksums are not optional here.** Serving the wrong weights produces
plausible predictions and no error, which is the worst failure mode this
system has. Every file is verified against the SHA-256 in the manifest before
the API is allowed to start, and a mismatch fails the init container, which
means the pod never serves.

Sources, chosen by URL scheme:

* ``s3://bucket/prefix`` - S3 or anything speaking its API (MinIO, R2, Spaces)
* ``https://host/path``  - any static host or a signed URL
* ``file:///path``       - a local directory, used by tests and local runs

Usage::

    python scripts/fetch_artifacts.py \\
        --source s3://models/mlcv/v1.2.0 \\
        --dest /models \\
        --manifest models/artifacts_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CHUNK = 1024 * 1024


@dataclass
class Artifact:
    name: str
    sha256: str
    size_bytes: int


class VerificationError(RuntimeError):
    """A downloaded file does not match its recorded checksum."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[Artifact]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Artifact(name=name, sha256=entry["sha256"], size_bytes=int(entry.get("size_bytes", 0)))
        for name, entry in sorted(payload.get("artifacts", {}).items())
    ]


def serving_artifacts(registry_path: Path) -> list[str]:
    """Filenames the registry actually needs to serve traffic.

    Derived from the registry rather than globbing the directory, so a
    training checkpoint or an engine sidecar left lying around does not end up
    as 96 MB every pod downloads and never opens.
    """
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    needed: set[str] = set()
    for entry in registry.get("models", []):
        needed.update((entry.get("artifacts") or {}).values())
        if entry.get("labels_file"):
            needed.add(entry["labels_file"])
    return sorted(needed)


def build_manifest(artifacts_dir: Path, names: list[str] | None = None) -> dict:
    """Record the checksum of every artifact, for the manifest file.

    Run this when the weights change. The manifest is what the init container
    verifies against, so it is the thing that decides which weights are
    allowed to serve.
    """
    entries = {}
    candidates = sorted(artifacts_dir.glob("*"))
    for path in candidates:
        if not path.is_file() or path.suffix not in (".onnx", ".json", ".pt", ".engine"):
            continue
        if names and path.name not in names:
            continue
        entries[path.name] = {
            "sha256": sha256_of(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": entries,
    }


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def _fetch_file(source: str, name: str, dest: Path) -> None:
    src = Path(urllib.parse.urlparse(source).path.lstrip("/")) / name
    if not src.exists():
        # On Windows the parsed path loses the drive letter, so fall back to
        # treating the whole thing as a plain path.
        src = Path(source.removeprefix("file://").lstrip("/")) / name
    shutil.copyfile(src, dest)


def _fetch_https(source: str, name: str, dest: Path) -> None:
    url = f"{source.rstrip('/')}/{name}"

    # Re-check the scheme on the assembled URL, not just the source. urlopen
    # happily handles file:// and ftp://, so a source that redirected or was
    # mis-configured could read a local path through what looks like an HTTP
    # fetch. Only http and https reach the opener.
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"refusing to fetch over {scheme!r}: only http and https")

    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, headers={"User-Agent": "mlcv-fetch-artifacts"}
    )
    with (
        urllib.request.urlopen(request, timeout=300) as response,  # noqa: S310
        dest.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, CHUNK)


def _fetch_s3(source: str, name: str, dest: Path) -> None:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on the image
        raise RuntimeError(
            "s3:// sources need boto3. Add it to the image, or use an https:// "
            "URL, which needs nothing."
        ) from exc

    parsed = urllib.parse.urlparse(source)
    bucket = parsed.netloc
    key = f"{parsed.path.strip('/')}/{name}".lstrip("/")

    # endpoint_url lets this talk to MinIO, R2 or Spaces, not only AWS.
    client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL") or None)
    client.download_file(bucket, key, str(dest))


FETCHERS = {"file": _fetch_file, "http": _fetch_https, "https": _fetch_https, "s3": _fetch_s3}


def fetch_one(source: str, artifact: Artifact, dest_dir: Path, *, retries: int = 3) -> Path:
    """Download one artifact and verify it. Raises unless it matches."""
    scheme = urllib.parse.urlparse(source).scheme or "file"
    fetcher = FETCHERS.get(scheme)
    if fetcher is None:
        raise ValueError(f"unsupported source scheme {scheme!r}: expected file, https or s3")

    final = dest_dir / artifact.name
    if final.exists() and sha256_of(final) == artifact.sha256:
        print(f"  {artifact.name}: already present and verified")
        return final

    # Download beside the target, then rename. A pod killed mid-download must
    # not leave a truncated file that looks complete to the next start.
    staging = dest_dir / f".{artifact.name}.partial"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            fetcher(source, artifact.name, staging)
            actual = sha256_of(staging)
            if actual != artifact.sha256:
                raise VerificationError(
                    f"{artifact.name} does not match the manifest.\n"
                    f"  expected sha256 {artifact.sha256}\n"
                    f"  got             {actual}\n"
                    "Refusing to serve it: wrong weights produce plausible "
                    "predictions and no error."
                )
            staging.replace(final)
            mb = artifact.size_bytes / 1e6
            print(f"  {artifact.name}: {mb:.1f} MB, checksum ok")
            return final
        except VerificationError:
            # A checksum mismatch is not transient. Retrying downloads the
            # same wrong bytes again.
            staging.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            staging.unlink(missing_ok=True)
            if attempt < retries:
                wait = 2**attempt
                print(f"  {artifact.name}: attempt {attempt} failed ({exc}), retrying in {wait}s")
                time.sleep(wait)

    raise RuntimeError(f"could not fetch {artifact.name} after {retries} attempts: {last_error}")


def fetch_all(*, source: str, dest: Path, manifest: Path, retries: int = 3) -> list[Path]:
    """Fetch and verify everything the manifest lists."""
    artifacts = load_manifest(manifest)
    if not artifacts:
        raise ValueError(f"{manifest} lists no artifacts")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetching {len(artifacts)} artifact(s) from {source} into {dest}")

    fetched = [fetch_one(source, a, dest, retries=retries) for a in artifacts]
    total = sum(p.stat().st_size for p in fetched) / 1e6
    print(f"all {len(fetched)} artifacts verified, {total:.1f} MB total")
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source",
        default=os.getenv("ARTIFACT_SOURCE"),
        help="s3://bucket/prefix, https://host/path, or file:///path",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.getenv("ARTIFACT_DEST", "/models")),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "models" / "artifacts_manifest.json",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Record checksums for the artifacts in --dest and write them here, "
        "instead of fetching. Run this when the weights change.",
    )
    args = parser.parse_args()

    if args.write_manifest:
        # Only what the registry serves. Including everything in the directory
        # made the manifest 503 MB against 388 MB, the difference being a
        # training checkpoint no pod ever opens.
        names = serving_artifacts(REPO_ROOT / "models" / "registry.json")
        manifest = build_manifest(args.dest, names)
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"wrote {args.write_manifest} with {len(manifest['artifacts'])} artifact(s)")
        return 0

    if not args.source:
        print(
            "error: no --source and no ARTIFACT_SOURCE.\n"
            "Nothing to fetch from, and starting without weights would give a "
            "pod that reports healthy and cannot predict.",
            file=sys.stderr,
        )
        return 2

    if not args.manifest.is_file():
        print(f"error: no manifest at {args.manifest}", file=sys.stderr)
        return 2

    try:
        fetch_all(source=args.source, dest=args.dest, manifest=args.manifest, retries=args.retries)
    except VerificationError as exc:
        print(f"\nCHECKSUM MISMATCH\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nfetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = ["Artifact", "VerificationError", "build_manifest", "fetch_all", "fetch_one", "sha256_of"]

if __name__ == "__main__":
    raise SystemExit(main())
