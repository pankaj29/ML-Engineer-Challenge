"""Artifact fetching for the init container.

The property that matters is the refusal. Serving the wrong weights produces
plausible predictions and no error anywhere, so a checksum mismatch has to
stop the pod rather than log a warning and carry on.
"""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path

import pytest

from scripts.fetch_artifacts import (
    Artifact,
    VerificationError,
    _file_url_to_path,
    build_manifest,
    fetch_all,
    fetch_one,
    serving_artifacts,
    sha256_of,
)


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """A directory standing in for the artifact store."""
    src = tmp_path / "store"
    src.mkdir()
    (src / "model.onnx").write_bytes(b"pretend onnx bytes" * 100)
    (src / "labels.json").write_text('{"0": "cat"}', encoding="utf-8")
    return src


@pytest.fixture
def manifest(source_dir: Path, tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(build_manifest(source_dir)), encoding="utf-8")
    return path


class TestFetching:
    def test_files_arrive_and_verify(self, source_dir, manifest, tmp_path) -> None:
        dest = tmp_path / "models"
        fetched = fetch_all(source=source_dir.as_uri(), dest=dest, manifest=manifest)

        assert len(fetched) == 2
        assert (dest / "model.onnx").read_bytes() == (source_dir / "model.onnx").read_bytes()

    def test_an_already_present_file_is_not_downloaded_again(
        self, source_dir, manifest, tmp_path
    ) -> None:
        """Pod restarts are common. Re-downloading 380 MB each time would add
        minutes to every restart."""
        dest = tmp_path / "models"
        fetch_all(source=source_dir.as_uri(), dest=dest, manifest=manifest)

        # Break the source. A second run must succeed from what is on disk.
        (source_dir / "model.onnx").write_bytes(b"different")
        fetch_all(source=source_dir.as_uri(), dest=dest, manifest=manifest)
        assert (
            sha256_of(dest / "model.onnx")
            == json.loads(manifest.read_text())["artifacts"]["model.onnx"]["sha256"]
        )

    def test_a_stale_local_file_is_replaced(self, source_dir, manifest, tmp_path) -> None:
        """Present but wrong is worse than absent, so it must not be kept."""
        dest = tmp_path / "models"
        dest.mkdir()
        (dest / "model.onnx").write_bytes(b"stale content from an older release")

        fetch_all(source=source_dir.as_uri(), dest=dest, manifest=manifest)
        assert (dest / "model.onnx").read_bytes() == (source_dir / "model.onnx").read_bytes()


class TestTheChecksumGate:
    def test_a_corrupted_artifact_is_refused(self, source_dir, manifest, tmp_path) -> None:
        entry = json.loads(manifest.read_text())["artifacts"]["model.onnx"]
        artifact = Artifact("model.onnx", entry["sha256"], entry["size_bytes"])

        # Same name, different bytes: a bad upload, a truncated transfer, or
        # the wrong release in the bucket.
        (source_dir / "model.onnx").write_bytes(b"corrupted")

        with pytest.raises(VerificationError, match="does not match the manifest"):
            fetch_one(source_dir.as_uri(), artifact, tmp_path, retries=3)

    def test_a_refused_artifact_leaves_nothing_behind(self, source_dir, manifest, tmp_path) -> None:
        """A half-written file would be picked up as valid by the next start."""
        entry = json.loads(manifest.read_text())["artifacts"]["model.onnx"]
        artifact = Artifact("model.onnx", entry["sha256"], entry["size_bytes"])
        (source_dir / "model.onnx").write_bytes(b"corrupted")

        with pytest.raises(VerificationError):
            fetch_one(source_dir.as_uri(), artifact, tmp_path)

        assert not (tmp_path / "model.onnx").exists()
        assert list(tmp_path.glob(".*partial")) == []

    def test_a_mismatch_is_not_retried(self, source_dir, tmp_path, monkeypatch) -> None:
        """Retrying downloads the same wrong bytes. Only transport errors
        deserve a retry."""
        calls = {"n": 0}
        real = Path(source_dir / "model.onnx").read_bytes

        def counting_copy(src, dst):
            calls["n"] += 1
            Path(dst).write_bytes(b"always wrong")

        monkeypatch.setattr("scripts.fetch_artifacts.shutil.copyfile", counting_copy)
        artifact = Artifact("model.onnx", sha256_of(source_dir / "model.onnx"), len(real()))

        with pytest.raises(VerificationError):
            fetch_one(source_dir.as_uri(), artifact, tmp_path, retries=5)
        assert calls["n"] == 1, f"retried a checksum failure {calls['n']} times"


class TestOverHttp:
    """The realistic source is object storage over HTTPS."""

    def test_a_real_http_download_verifies(self, source_dir, manifest, tmp_path) -> None:
        handler = http.server.SimpleHTTPRequestHandler

        class Quiet(handler):
            def log_message(self, *args):
                return

            def translate_path(self, path):
                return str(source_dir / path.lstrip("/").split("?")[0])

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            fetched = fetch_all(
                source=f"http://127.0.0.1:{port}",
                dest=tmp_path / "models",
                manifest=manifest,
            )
            assert len(fetched) == 2
        finally:
            server.shutdown()

    def test_a_non_http_scheme_is_refused_by_the_http_fetcher(self, tmp_path) -> None:
        """urlopen handles file:// too, so an https source that somehow names
        a local path must not read it."""
        from scripts.fetch_artifacts import _fetch_https

        with pytest.raises(ValueError, match="only http and https"):
            _fetch_https("ftp://example.com", "model.onnx", tmp_path / "out")


class TestManifestContents:
    def test_only_what_the_registry_serves_is_listed(self, tmp_path) -> None:
        """A training checkpoint in the artifacts directory is 96 MB that
        every pod would download and never open."""
        registry = tmp_path / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "name": "m",
                            "artifacts": {"onnx": "m.onnx"},
                            "labels_file": "labels.json",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert serving_artifacts(registry) == ["labels.json", "m.onnx"]

    def test_the_repo_manifest_matches_the_files_on_disk(self) -> None:
        """If this fails the weights changed and the manifest was not
        regenerated, which would fail every pod start."""
        root = Path(__file__).resolve().parents[2]
        path = root / "models" / "artifacts_manifest.json"
        if not path.is_file():
            pytest.skip("no manifest committed")

        manifest = json.loads(path.read_text(encoding="utf-8"))
        artifacts = root / "models" / "artifacts"
        for name, entry in manifest["artifacts"].items():
            local = artifacts / name
            if not local.is_file():
                pytest.skip(f"{name} is not checked out")
            # Size is the wrong test: a labels file is legitimately tiny. An
            # unfetched LFS file is identifiable by its contents.
            if local.open("rb").read(42).startswith(b"version https://git-lfs"):
                pytest.skip(f"{name} is an unfetched LFS pointer")
            assert sha256_of(local) == entry["sha256"], (
                f"{name} on disk does not match the manifest; regenerate with "
                "python scripts/fetch_artifacts.py --dest models/artifacts "
                "--write-manifest models/artifacts_manifest.json"
            )


class TestFileUrlParsing:
    """Parsed directly, because the round-trip tests above cannot catch this.

    They build their URL from the platform they run on, so a parser that only
    works on Windows passes the whole suite on Windows. That is exactly what
    happened: `.lstrip("/")` handled the `/C:/...` the parser returns for a
    Windows drive, and silently turned every absolute POSIX path into a
    relative one. Green locally, five failures on the Linux runner with
    "No such file or directory: 'tmp/...'".

    These cases are fixed strings, so both shapes are checked everywhere.
    """

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # POSIX absolute. The leading slash is the path, not padding.
            ("file:///tmp/pytest-0/store", "/tmp/pytest-0/store"),
            ("file:///artifacts", "/artifacts"),
            ("file:///var/lib/models/v2", "/var/lib/models/v2"),
            # Windows, three-slash, which is what Path.as_uri() emits.
            ("file:///C:/Users/x/models", "C:/Users/x/models"),
            ("file:///D:/artifacts", "D:/artifacts"),
            # Windows, two-slash. Malformed but commonly written by hand.
            ("file://C:/Users/x/models", "C:/Users/x/models"),
            # Percent-encoding, because paths have spaces in them.
            ("file:///path%20with%20space/a", "/path with space/a"),
            # localhost is the spec's way of saying "this machine".
            ("file://localhost/srv/models", "/srv/models"),
        ],
    )
    def test_urls_resolve_the_same_way_on_any_platform(self, url: str, expected: str) -> None:
        assert _file_url_to_path(url).as_posix() == expected

    def test_a_posix_path_never_comes_back_relative(self) -> None:
        """The specific failure. A relative path resolves against the working
        directory, which in a container is not where the artifacts are."""
        result = _file_url_to_path("file:///tmp/store")
        assert result.as_posix().startswith("/"), f"{result} is relative"

    def test_a_real_host_is_treated_as_a_unc_path(self) -> None:
        assert _file_url_to_path("file://fileserver/share/models").as_posix() == (
            "//fileserver/share/models"
        )
