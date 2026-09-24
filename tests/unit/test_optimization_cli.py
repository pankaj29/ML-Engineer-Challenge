"""Unit tests for the optimisation command-line entry points.

The `main()` functions were the bulk of what remained uncovered in
`models/optimization/*`. They are worth testing rather than waving through:
each is the interface a human or a CI job actually uses, and each writes a
report that later gets quoted as a measurement.

Every test drives the real `main()` with a patched `sys.argv`, against a tiny
model, and asserts on the file it produces.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.optimization import (
    benchmark as benchmark_mod,
    export_onnx as export_mod,
    quantize as quantize_mod,
)
from models.optimization.benchmark import benchmark_torch
from models.optimization.export_onnx import export_to_onnx
from models.optimization.quantize import quantize_torch_dynamic

torch = pytest.importorskip("torch")

INPUT_SHAPE = (1, 3, 16, 16)


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 16 * 16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    ).eval()


@pytest.fixture
def onnx_file(tmp_path: Path) -> Path:
    path = tmp_path / "cli.onnx"
    export_to_onnx(_tiny_model(), path, input_shape=INPUT_SHAPE, name="cli")
    return path


@pytest.fixture
def calibration_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "calib"
    directory.mkdir()
    rng = np.random.default_rng(3)
    for i in range(4):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(buf, format="PNG")
        (directory / f"{i}.png").write_bytes(buf.getvalue())
    return directory


def _run(module, argv: list[str], monkeypatch) -> int:
    monkeypatch.setattr("sys.argv", [module.__name__, *argv])
    return module.main()


class TestBenchmarkCli:
    def test_benchmarks_one_model_and_writes_a_report(
        self, onnx_file: Path, tmp_path: Path, monkeypatch
    ) -> None:
        out = tmp_path / "reports"
        code = _run(
            benchmark_mod,
            [
                "--onnx", str(onnx_file),
                "--batch-sizes", "1",
                "--iterations", "3",
                "--warmup", "1",
                "--image-size", "16",
                "--output-dir", str(out),
            ],
            monkeypatch,
        )
        assert code == 0
        written = list(out.glob("*"))
        assert written, "no report written"

    def test_report_is_valid_json(self, onnx_file: Path, tmp_path: Path, monkeypatch) -> None:
        out = tmp_path / "reports"
        _run(
            benchmark_mod,
            ["--onnx", str(onnx_file), "--batch-sizes", "1", "--iterations", "3",
             "--warmup", "1", "--image-size", "16", "--output-dir", str(out)],
            monkeypatch,
        )
        for path in out.glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))

    def test_report_includes_a_markdown_table(
        self, onnx_file: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """The markdown is what ends up quoted in the docs."""
        out = tmp_path / "reports"
        _run(
            benchmark_mod,
            ["--onnx", str(onnx_file), "--batch-sizes", "1", "--iterations", "3",
             "--warmup", "1", "--image-size", "16", "--output-dir", str(out)],
            monkeypatch,
        )
        markdown = list(out.glob("*.md"))
        assert markdown
        assert "|" in markdown[0].read_text(encoding="utf-8")

    def test_several_batch_sizes(self, onnx_file: Path, tmp_path: Path, monkeypatch) -> None:
        out = tmp_path / "reports"
        assert (
            _run(
                benchmark_mod,
                ["--onnx", str(onnx_file), "--batch-sizes", "1,2", "--iterations", "2",
                 "--warmup", "1", "--image-size", "16", "--output-dir", str(out)],
                monkeypatch,
            )
            == 0
        )

    def test_missing_model_is_reported_not_crashed(self, tmp_path: Path, monkeypatch) -> None:
        code = _run(
            benchmark_mod,
            ["--onnx", str(tmp_path / "absent.onnx"), "--iterations", "2",
             "--output-dir", str(tmp_path / "r")],
            monkeypatch,
        )
        assert code != 0


    def test_no_models_anywhere_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        code = _run(
            benchmark_mod,
            ["--artifacts", str(empty), "--output-dir", str(tmp_path / "r")],
            monkeypatch,
        )
        assert code == 2

    def test_device_auto_resolves_without_a_gpu(
        self, onnx_file: Path, tmp_path: Path, monkeypatch
    ) -> None:
        code = _run(
            benchmark_mod,
            ["--onnx", str(onnx_file), "--device", "auto", "--batch-sizes", "1",
             "--iterations", "2", "--warmup", "1", "--image-size", "16",
             "--output-dir", str(tmp_path / "r")],
            monkeypatch,
        )
        assert code == 0


class TestBenchmarkHonestyAboutDevice:
    def test_asking_for_cuda_without_cuda_warns_loudly(self, onnx_file: Path) -> None:
        """A CPU number filed under "cuda" would be quoted as a GPU result."""
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            pytest.skip("this machine actually has CUDA")

        with pytest.warns(RuntimeWarning, match="CUDA was requested"):
            results = benchmark_mod.benchmark_onnx(
                onnx_file, input_shape=(3, 16, 16), batch_sizes=(1,),
                iterations=2, warmup=1, device="cuda",
            )
        assert results[0].device == "cpu"

    def test_a_batch_that_fails_is_recorded_not_fatal(self, onnx_file: Path) -> None:
        """One bad batch size must not discard the other measurements."""
        results = benchmark_mod.benchmark_onnx(
            onnx_file, input_shape=(3, 99, 99), batch_sizes=(1,), iterations=2, warmup=1
        )
        assert results
        assert any("failed at batch" in n for n in results[0].notes)


class TestRegistryInputShapes:
    def test_returns_a_mapping_of_model_stem_to_shape(self) -> None:
        shapes = benchmark_mod._registry_input_shapes()
        assert isinstance(shapes, dict)
        assert all(isinstance(v, tuple) for v in shapes.values())

    def test_an_unreadable_registry_is_not_fatal(self, monkeypatch) -> None:
        """Benchmarking loose .onnx files must work with no registry at all."""
        import api.services.model_service as ms

        monkeypatch.setattr(
            ms, "ModelService", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no registry"))
        )
        assert benchmark_mod._registry_input_shapes() == {}


class TestBenchmarkTorch:
    def test_benchmarks_a_torch_module(self) -> None:
        results = benchmark_torch(
            _tiny_model(), name="tiny", input_shape=(3, 16, 16),
            batch_sizes=(1,), iterations=3, warmup=1,
        )
        assert results
        assert results[0].runtime.startswith("torch")
        assert results[0].throughput_ips > 0

    def test_reports_the_requested_batch_sizes(self) -> None:
        results = benchmark_torch(
            _tiny_model(), name="tiny", input_shape=(3, 16, 16),
            batch_sizes=(1, 2), iterations=2, warmup=1,
        )
        assert {r.batch_size for r in results} == {1, 2}


class TestMarkdownComparisonTable:
    """The speedup/shrink table is the part a reader actually acts on."""

    @staticmethod
    def _result(runtime: str, p50: float, size: float, notes=()):
        return benchmark_mod.BenchmarkResult(
            name="tiny", runtime=runtime, device="cpu", batch_size=1, iterations=10,
            mean_ms=p50, median_ms=p50, p50_ms=p50, p90_ms=p50, p95_ms=p50, p99_ms=p50,
            min_ms=p50, max_ms=p50, stdev_ms=0.0,
            throughput_ips=1000.0 / p50, size_mb=size, notes=list(notes),
        )

    def test_int8_is_compared_against_the_fp32_baseline(self) -> None:
        markdown = benchmark_mod.render_markdown(
            [self._result("onnx", 4.0, 100.0), self._result("onnx_int8", 2.0, 25.0)], {}
        )
        assert "2.00x" in markdown, "speedup column missing"
        assert "4.00x" in markdown, "size-shrink column missing"

    def test_notes_are_surfaced_in_the_report(self) -> None:
        markdown = benchmark_mod.render_markdown(
            [self._result("onnx", 4.0, 100.0, notes=["accuracy NOT verified"])], {}
        )
        assert "## Notes" in markdown
        assert "accuracy NOT verified" in markdown


class TestExportCli:
    """`--output` names a *directory*; the filename comes from `--model`."""

    def test_exports_a_torchvision_model(self, tmp_path: Path, monkeypatch) -> None:
        out = tmp_path / "exported"
        code = _run(
            export_mod,
            ["--model", "resnet18", "--image-size", "32", "--output", str(out),
             "--report", str(tmp_path / "export.json")],
            monkeypatch,
        )
        assert code == 0
        assert (out / "resnet18.onnx").is_file()

    def test_writes_a_verification_report(self, tmp_path: Path, monkeypatch) -> None:
        report = tmp_path / "export.json"
        _run(
            export_mod,
            ["--model", "resnet18", "--image-size", "32",
             "--output", str(tmp_path / "m"), "--report", str(report)],
            monkeypatch,
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload = payload[0] if isinstance(payload, list) else payload
        assert payload["verified"] is True
        assert payload["max_abs_diff"] < 1e-3

    def test_num_classes_swaps_the_head(self, tmp_path: Path, monkeypatch) -> None:
        """The fine-tuned model has 200 Tiny-ImageNet classes, not 1000."""
        report = tmp_path / "export.json"
        code = _run(
            export_mod,
            ["--model", "resnet18", "--image-size", "32", "--num-classes", "200",
             "--output", str(tmp_path / "m"), "--report", str(report)],
            monkeypatch,
        )
        assert code == 0
        assert json.loads(report.read_text(encoding="utf-8"))["output_shapes"][0][-1] == 200

    def test_loads_a_checkpoint(self, tmp_path: Path, monkeypatch) -> None:
        """The path that actually ships: trained weights, not ImageNet ones."""
        import torchvision.models as tvm

        model = tvm.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 200)
        checkpoint = tmp_path / "ckpt.pt"
        torch.save({"model_state_dict": model.state_dict()}, checkpoint)

        code = _run(
            export_mod,
            ["--model", "resnet18", "--image-size", "32", "--num-classes", "200",
             "--checkpoint", str(checkpoint),
             "--output", str(tmp_path / "m"), "--report", str(tmp_path / "r.json")],
            monkeypatch,
        )
        assert code == 0

    def test_unknown_architecture_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        code = _run(
            export_mod,
            ["--model", "not-a-real-architecture", "--output", str(tmp_path / "x")],
            monkeypatch,
        )
        assert code != 0


class TestExportVerificationNotes:
    def test_an_impossible_tolerance_is_recorded_as_a_mismatch(self, tmp_path: Path) -> None:
        """`verified` must be driven by the numbers, not by "the export ran"."""
        result = export_to_onnx(
            _tiny_model(), tmp_path / "strict.onnx",
            input_shape=INPUT_SHAPE, name="strict", tolerance=0.0,
        )
        assert result.verified is False
        assert any("NUMERICAL MISMATCH" in n for n in result.notes)

    def test_a_structurally_broken_graph_is_noted_not_hidden(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import onnx

        monkeypatch.setattr(
            onnx.checker, "check_model", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
        )
        result = export_to_onnx(
            _tiny_model(), tmp_path / "broken.onnx", input_shape=INPUT_SHAPE, name="broken"
        )
        assert any("structural check failed" in n for n in result.notes)


class TestQuantizeCli:
    def test_dynamic_mode(self, onnx_file: Path, tmp_path: Path, monkeypatch) -> None:
        code = _run(
            quantize_mod,
            ["--onnx", str(onnx_file), "--mode", "dynamic",
             "--report", str(tmp_path / "q.json")],
            monkeypatch,
        )
        assert code == 0
        assert (tmp_path / "q.json").exists()

    def test_static_mode(
        self, onnx_file: Path, calibration_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        code = _run(
            quantize_mod,
            ["--onnx", str(onnx_file), "--mode", "static",
             "--calibration-dir", str(calibration_dir), "--num-calibration", "3",
             "--image-size", "16", "--report", str(tmp_path / "qs.json")],
            monkeypatch,
        )
        assert code == 0

    def test_both_modes_in_one_run(
        self, onnx_file: Path, calibration_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        report = tmp_path / "qb.json"
        code = _run(
            quantize_mod,
            ["--onnx", str(onnx_file), "--mode", "both",
             "--calibration-dir", str(calibration_dir), "--num-calibration", "3",
             "--image-size", "16", "--report", str(report)],
            monkeypatch,
        )
        assert code == 0
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert len(payload) >= 2, "expected one entry per mode"

    def test_static_without_calibration_data_is_refused(
        self, onnx_file: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Static quantization without calibration is not a thing."""
        code = _run(
            quantize_mod,
            ["--onnx", str(onnx_file), "--mode", "static",
             "--calibration-dir", str(tmp_path / "absent")],
            monkeypatch,
        )
        assert code != 0

    def test_missing_model_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        code = _run(
            quantize_mod, ["--onnx", str(tmp_path / "absent.onnx"), "--mode", "dynamic"], monkeypatch
        )
        assert code != 0


class TestQuantizeTorchDynamic:
    def test_produces_a_smaller_torchscript_archive(self, tmp_path: Path) -> None:
        result = quantize_torch_dynamic(
            _tiny_model(), tmp_path / "q.pt", input_shape=INPUT_SHAPE, name="tiny"
        )
        assert Path(result.output_path).exists()
        assert result.compression_ratio >= 1.0
        assert "torch" in result.mode
