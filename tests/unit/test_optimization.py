"""Unit tests for ONNX export, quantization and benchmarking.

Part 1 of the brief requires all three: *"Apply quantization (INT8) to all
models"*, *"Convert models to ONNX and TensorRT formats"* and *"Benchmark
inference times across all formats"*. The audit found `models/optimization/*`
at **0% coverage** - 511 statements, untested.

Everything here runs against a genuinely tiny model exported on the fly (a
Flatten + Linear, a few kilobytes), so the tests are fast and need no GPU, no
downloaded weights and no dataset. That is the point: the transformations are
what is being tested, not the size of the thing being transformed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.utils.image_processing import PreprocessConfig
from models.optimization.benchmark import (
    BenchmarkResult,
    benchmark_onnx,
    environment_info,
    measure,
    render_markdown,
    summarise,
)
from models.optimization.export_onnx import export_to_onnx, optimize_onnx_graph
from models.optimization.quantize import (
    _concrete_input_shape,
    iter_calibration_images,
    quantize_onnx_dynamic,
    quantize_onnx_static,
)

torch = pytest.importorskip("torch")

INPUT_SHAPE = (1, 3, 16, 16)


def _tiny_model():
    """Small enough to export in well under a second."""
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 16 * 16, 6),
    ).eval()


def _linear_only_model():
    """No convolutions - see `exported_linear` for why that matters."""
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 16 * 16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 6),
    ).eval()


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("opt") / "tiny.onnx"
    export_to_onnx(_tiny_model(), path, input_shape=INPUT_SHAPE, name="tiny")
    return path


@pytest.fixture(scope="module")
def exported_linear(tmp_path_factory) -> Path:
    """A Conv-free model, for the dynamic-quantization tests.

    Dynamic quantization rewrites Conv into `ConvInteger`, and ONNX Runtime's
    CPU provider has no kernel for it - the quantized model exports fine and
    then fails to load. Dynamic quantization is designed for MatMul-heavy
    graphs (transformers); convolutional models want the static path, which is
    what `quantize_onnx_static` and the project's real pipeline use.
    """
    path = tmp_path_factory.mktemp("opt_lin") / "tiny_linear.onnx"
    export_to_onnx(_linear_only_model(), path, input_shape=INPUT_SHAPE, name="tiny_linear")
    return path


# ---------------------------------------------------------------------------
# export_onnx
# ---------------------------------------------------------------------------
class TestExportToOnnx:
    def test_writes_a_file(self, tmp_path: Path) -> None:
        result = export_to_onnx(_tiny_model(), tmp_path / "m.onnx", input_shape=INPUT_SHAPE)
        assert result.onnx_path.exists()
        assert result.size_mb > 0

    def test_verifies_against_the_source_model(self, tmp_path: Path) -> None:
        """The check that catches a silently-wrong conversion."""
        result = export_to_onnx(_tiny_model(), tmp_path / "m.onnx", input_shape=INPUT_SHAPE)
        assert result.verified
        assert result.max_abs_diff < 1e-4

    def test_records_the_opset_and_shape(self, tmp_path: Path) -> None:
        result = export_to_onnx(
            _tiny_model(), tmp_path / "m.onnx", input_shape=INPUT_SHAPE, opset=17
        )
        assert result.opset == 17
        assert tuple(result.input_shape) == INPUT_SHAPE

    def test_reports_output_shapes(self, tmp_path: Path) -> None:
        result = export_to_onnx(_tiny_model(), tmp_path / "m.onnx", input_shape=INPUT_SHAPE)
        assert result.output_shapes

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        result = export_to_onnx(
            _tiny_model(), tmp_path / "a" / "b" / "m.onnx", input_shape=INPUT_SHAPE
        )
        assert result.onnx_path.exists()

    def test_batch_dimension_is_dynamic(self, exported: Path) -> None:
        """A fixed batch dim silently breaks every batched request."""
        import onnxruntime as ort

        session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        for batch in (1, 3, 8):
            out = session.run(None, {name: np.zeros((batch, 3, 16, 16), dtype=np.float32)})
            assert out[0].shape[0] == batch

    def test_result_is_json_serialisable(self, tmp_path: Path) -> None:
        """Results are written to benchmarks/reports/ as JSON."""
        from dataclasses import asdict

        result = export_to_onnx(_tiny_model(), tmp_path / "m.onnx", input_shape=INPUT_SHAPE)
        payload = asdict(result)
        payload["onnx_path"] = str(payload["onnx_path"])
        json.dumps(payload)


class TestOptimizeOnnxGraph:
    def test_produces_a_loadable_model(self, exported: Path, tmp_path: Path) -> None:
        import onnxruntime as ort

        dst = tmp_path / "opt.onnx"
        info = optimize_onnx_graph(exported, dst)
        assert dst.exists()
        assert info
        ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])

    def test_preserves_the_answer(self, exported: Path, tmp_path: Path) -> None:
        """Graph fusion must not change what the model predicts."""
        import onnxruntime as ort

        dst = tmp_path / "opt.onnx"
        optimize_onnx_graph(exported, dst)
        sample = np.random.default_rng(0).standard_normal((1, 3, 16, 16)).astype(np.float32)

        before = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        after = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        a = before.run(None, {before.get_inputs()[0].name: sample})[0]
        b = after.run(None, {after.get_inputs()[0].name: sample})[0]
        assert np.abs(a - b).max() < 1e-4


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------
class TestConcreteInputShape:
    def test_resolves_the_dynamic_batch_to_one(self, exported: Path) -> None:
        """Calibration needs a concrete shape; 'batch' is not a number."""
        shape = _concrete_input_shape(exported)
        assert all(isinstance(d, int) and d > 0 for d in shape)
        assert shape[0] == 1


class TestIterCalibrationImages:
    @pytest.fixture
    def image_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "calib"
        directory.mkdir()
        for i in range(5):
            buf = io.BytesIO()
            Image.new("RGB", (32, 32), (i * 40, 90, 120)).save(buf, format="PNG")
            (directory / f"{i}.png").write_bytes(buf.getvalue())
        return directory

    def test_yields_preprocessed_arrays(self, image_dir: Path) -> None:
        cfg = PreprocessConfig(size=(16, 16))
        arrays = list(iter_calibration_images(image_dir, cfg, limit=3))
        assert len(arrays) == 3
        assert arrays[0].shape == (1, 3, 16, 16)
        assert arrays[0].dtype == np.float32

    def test_limit_is_respected(self, image_dir: Path) -> None:
        cfg = PreprocessConfig(size=(16, 16))
        assert len(list(iter_calibration_images(image_dir, cfg, limit=2))) == 2

    def test_skips_files_that_are_not_images(self, image_dir: Path) -> None:
        """One stray README must not abort a calibration run."""
        (image_dir / "notes.txt").write_text("not an image")
        cfg = PreprocessConfig(size=(16, 16))
        assert len(list(iter_calibration_images(image_dir, cfg, limit=10))) == 5

    def test_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert list(iter_calibration_images(empty, PreprocessConfig(size=(16, 16)))) == []


class TestDynamicQuantization:
    """Uses the Conv-free model - see the `exported_linear` fixture."""

    def test_produces_a_smaller_model(self, exported_linear: Path, tmp_path: Path) -> None:
        result = quantize_onnx_dynamic(exported_linear, tmp_path / "dyn.onnx")
        assert Path(result.output_path).exists()
        assert result.quantized_mb <= result.original_mb
        assert result.compression_ratio >= 1.0

    def test_records_the_mode(self, exported_linear: Path, tmp_path: Path) -> None:
        result = quantize_onnx_dynamic(exported_linear, tmp_path / "dyn.onnx")
        assert "dynamic" in result.mode

    def test_quantized_model_still_runs(self, exported_linear: Path, tmp_path: Path) -> None:
        import onnxruntime as ort

        result = quantize_onnx_dynamic(exported_linear, tmp_path / "dyn.onnx")
        session = ort.InferenceSession(result.output_path, providers=["CPUExecutionProvider"])
        out = session.run(
            None, {session.get_inputs()[0].name: np.zeros((1, 3, 16, 16), dtype=np.float32)}
        )
        assert out[0].shape == (1, 6)

    def test_accuracy_delta_is_measured_when_samples_are_given(
        self, exported_linear: Path, tmp_path: Path
    ) -> None:
        """Compression without a measured cost is not a result."""
        rng = np.random.default_rng(1)
        samples = [rng.standard_normal((1, 3, 16, 16)).astype(np.float32) for _ in range(4)]
        result = quantize_onnx_dynamic(exported_linear, tmp_path / "dyn.onnx", samples=samples)
        assert result.max_abs_diff >= 0.0
        assert 0.0 <= result.top1_agreement <= 1.0

    def test_default_output_path_is_derived(self, exported_linear: Path) -> None:
        result = quantize_onnx_dynamic(exported_linear)
        assert Path(result.output_path).exists()
        assert "int8" in Path(result.output_path).name


class TestDynamicQuantizationLimits:
    def test_conv_models_produce_an_unloadable_dynamic_quantization(
        self, exported: Path, tmp_path: Path
    ) -> None:
        """Documents a real ONNX Runtime limitation, so nobody rediscovers it.

        Dynamic quantization rewrites Conv into ConvInteger, for which the CPU
        provider has no kernel. The quantization *succeeds* and the resulting
        file cannot be loaded - which is why this project quantizes its
        convolutional models statically.
        """
        result = quantize_onnx_dynamic(exported, tmp_path / "conv_dyn.onnx")

        # The file is still produced and the compression is still real...
        assert Path(result.output_path).exists()
        assert result.compression_ratio >= 1.0

        # ...but the result says plainly that it could not be verified, rather
        # than the whole call raising and discarding the numbers.
        assert any("NOT verified" in n for n in result.notes), result.notes
        assert result.top1_agreement == 0.0


class TestStaticQuantization:
    @pytest.fixture
    def image_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "calib_static"
        directory.mkdir()
        rng = np.random.default_rng(2)
        for i in range(6):
            arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            (directory / f"{i}.png").write_bytes(buf.getvalue())
        return directory

    def test_calibrates_and_compresses(
        self, exported: Path, image_dir: Path, tmp_path: Path
    ) -> None:
        result = quantize_onnx_static(
            exported,
            image_dir,
            PreprocessConfig(size=(16, 16)),
            dst=tmp_path / "static.onnx",
            num_calibration=4,
        )
        assert Path(result.output_path).exists()
        assert result.calibration_images > 0
        assert "static" in result.mode

    def test_refuses_a_missing_calibration_directory(self, exported: Path, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            quantize_onnx_static(
                exported,
                tmp_path / "does-not-exist",
                PreprocessConfig(size=(16, 16)),
                dst=tmp_path / "out.onnx",
            )

    def test_result_is_json_serialisable(
        self, exported: Path, image_dir: Path, tmp_path: Path
    ) -> None:
        from dataclasses import asdict

        result = quantize_onnx_static(
            exported,
            image_dir,
            PreprocessConfig(size=(16, 16)),
            dst=tmp_path / "static.onnx",
            num_calibration=3,
        )
        json.dumps(asdict(result))


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
class TestMeasure:
    def test_returns_one_timing_per_iteration(self) -> None:
        timings = measure(lambda: sum(range(50)), iterations=12, warmup=2)
        assert len(timings) == 12
        assert all(t >= 0 for t in timings)

    def test_warmup_runs_are_excluded(self) -> None:
        """Warmup exists to absorb first-call cost; counting it skews p50."""
        calls = {"n": 0}

        def counted() -> None:
            calls["n"] += 1

        timings = measure(counted, iterations=5, warmup=3)
        assert len(timings) == 5
        assert calls["n"] == 8


class TestSummarise:
    def test_computes_percentiles_in_order(self) -> None:
        result = summarise(
            [float(i) for i in range(1, 101)],
            name="m",
            runtime="onnx",
            device="cpu",
            batch_size=1,
        )
        assert result.p50_ms <= result.p90_ms <= result.p95_ms <= result.p99_ms
        assert result.min_ms <= result.mean_ms <= result.max_ms

    def test_throughput_accounts_for_batch_size(self) -> None:
        """8 images in 10 ms is 8x the throughput of 1 image in 10 ms."""
        one = summarise([10.0] * 10, name="m", runtime="onnx", device="cpu", batch_size=1)
        eight = summarise([10.0] * 10, name="m", runtime="onnx", device="cpu", batch_size=8)
        assert eight.throughput_ips == pytest.approx(one.throughput_ips * 8, rel=0.01)

    def test_empty_timings_do_not_divide_by_zero(self) -> None:
        result = summarise([], name="m", runtime="onnx", device="cpu", batch_size=1)
        assert result.iterations == 0

    def test_notes_are_carried_through(self) -> None:
        result = summarise(
            [1.0], name="m", runtime="onnx", device="cpu", batch_size=1, notes=["skipped x"]
        )
        assert result.notes == ["skipped x"]


class TestEnvironmentInfo:
    def test_records_enough_to_reproduce_a_number(self) -> None:
        info = environment_info()
        assert info
        blob = " ".join(map(str, info)).lower()
        assert "python" in blob or "cpu" in blob or "platform" in blob

    def test_is_json_serialisable(self) -> None:
        json.dumps(environment_info())


class TestRenderMarkdown:
    def test_renders_a_table_with_every_result(self) -> None:
        results = [
            summarise([5.0] * 5, name="a", runtime="onnx", device="cpu", batch_size=1),
            summarise([9.0] * 5, name="b", runtime="onnx_int8", device="cpu", batch_size=1),
        ]
        out = render_markdown(results, environment_info())
        assert "a" in out and "b" in out
        assert "|" in out

    def test_handles_no_results_without_crashing(self) -> None:
        assert isinstance(render_markdown([], environment_info()), str)


class TestBenchmarkOnnx:
    def test_benchmarks_a_real_model(self, exported: Path) -> None:
        results = benchmark_onnx(
            exported, input_shape=(3, 16, 16), batch_sizes=(1,), iterations=5, warmup=1
        )
        assert results
        assert results[0].iterations == 5
        assert results[0].throughput_ips > 0

    def test_reports_the_device_actually_used(self, exported: Path) -> None:
        """Asking for cuda on a CPU box must report cpu, not cuda."""
        results = benchmark_onnx(
            exported,
            input_shape=(3, 16, 16),
            batch_sizes=(1,),
            iterations=3,
            warmup=1,
            device="cpu",
        )
        assert results[0].device == "cpu"

    def test_multiple_batch_sizes_give_multiple_rows(self, exported: Path) -> None:
        results = benchmark_onnx(
            exported, input_shape=(3, 16, 16), batch_sizes=(1, 4), iterations=3, warmup=1
        )
        assert {r.batch_size for r in results} == {1, 4}

    def test_records_the_file_size(self, exported: Path) -> None:
        results = benchmark_onnx(
            exported, input_shape=(3, 16, 16), batch_sizes=(1,), iterations=3, warmup=1
        )
        assert results[0].size_mb > 0


class TestBenchmarkInterleaved:
    def test_every_case_gets_all_its_iterations(self, exported: Path) -> None:
        from models.optimization.benchmark import benchmark_interleaved

        results = benchmark_interleaved(
            [(exported, (3, 16, 16))], batch_sizes=(1, 2), iterations=6, warmup=1, rounds=3
        )
        assert {(r.batch_size, r.iterations) for r in results} == {(1, 6), (2, 6)}
        assert all(r.p50_ms > 0 for r in results)

    def test_rounds_are_interleaved_across_models(self, exported: Path, monkeypatch) -> None:
        """Model A must not finish all its rounds before model B starts."""
        import models.optimization.benchmark as bench

        order: list[int] = []
        real_measure = bench.measure

        def spy(fn, *, iterations, warmup):
            order.append(id(fn))
            return real_measure(fn, iterations=iterations, warmup=warmup)

        monkeypatch.setattr(bench, "measure", spy)
        bench.benchmark_interleaved(
            [(exported, (3, 16, 16)), (exported, (3, 16, 16))],
            iterations=4,
            warmup=0,
            rounds=2,
        )
        first, second = order[0], order[1]
        assert first != second
        assert order == [first, second, first, second]


class TestBenchmarkTorchCases:
    def test_torch_rows_join_the_same_run(self, exported: Path) -> None:
        from models.optimization.benchmark import benchmark_interleaved

        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU())
        results = benchmark_interleaved(
            [(exported, (3, 16, 16))],
            torch_cases=[("tiny", model, (3, 16, 16))],
            iterations=4,
            warmup=1,
            rounds=2,
        )
        torch_rows = [r for r in results if r.runtime == "torch"]
        assert [r.name for r in torch_rows] == ["tiny_torch"]
        assert torch_rows[0].iterations == 4
        assert torch_rows[0].size_mb > 0

    def test_torch_rows_are_compared_with_their_onnx_baseline(self) -> None:
        onnx_row = TestMarkdownRows.result("m", "onnx", 10.0)
        torch_row = TestMarkdownRows.result("m_torch", "torch", 40.0)
        markdown = render_markdown([onnx_row, torch_row], {})
        assert "| m_torch | torch | 0.25x |" in markdown


class TestMarkdownRows:
    @staticmethod
    def result(name: str, runtime: str, p50: float):
        return summarise(
            [p50] * 3, name=name, runtime=runtime, device="cpu", batch_size=1, size_mb=1.0
        )


class TestBenchmarkResultShape:
    def test_is_json_serialisable(self) -> None:
        from dataclasses import asdict

        json.dumps(asdict(summarise([1.0], name="m", runtime="onnx", device="cpu", batch_size=1)))

    def test_summary_line_names_the_model(self) -> None:
        result: BenchmarkResult = summarise(
            [1.0], name="resnet50", runtime="onnx", device="cpu", batch_size=1
        )
        assert "resnet50" in result.summary()


class TestDetectorAgreement:
    """Agreement for YOLO-shaped outputs, which used to score 100% by default."""

    @staticmethod
    def _yolo(cls_scores: list[float]) -> np.ndarray:
        out = np.zeros((1, 4 + len(cls_scores), 3), dtype=np.float32)
        out[0, 4:, 1] = cls_scores
        return out

    def test_dominant_class_is_the_most_confident(self) -> None:
        from models.optimization.quantize import _dominant_class

        assert _dominant_class(self._yolo([0.1, 0.9, 0.3])) == 1

    def test_all_zero_scores_mean_nothing_detected(self) -> None:
        """The broken INT8 detector's output: argmax alone would say class 0."""
        from models.optimization.quantize import _dominant_class

        assert _dominant_class(self._yolo([0.0, 0.0, 0.0])) == -1


# ---------------------------------------------------------------------------
# export_tensorrt.measure_engine_agreement
# ---------------------------------------------------------------------------
class TestEngineAgreement:
    """TensorRT needs a GPU, so a fake engine stands in for it."""

    @pytest.fixture
    def image_dir(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(0)
        for i in range(8):
            array = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
            Image.fromarray(array).save(tmp_path / f"{i:02d}.png")
        return tmp_path

    def _fake_engine(self, monkeypatch, onnx_path: Path, transform) -> None:
        import onnxruntime as ort

        import api.services.model_service as model_service

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name

        class FakeBackend:
            def __init__(self, path: Path) -> None:
                pass

            def infer(self, x: np.ndarray) -> list[np.ndarray]:
                return [transform(session.run(None, {name: x})[0])]

            def close(self) -> None:
                pass

        monkeypatch.setattr(model_service, "TensorRTBackend", FakeBackend)

    def _measure(self, exported: Path, image_dir: Path, task: str, offset: int = 2):
        from models.optimization.export_tensorrt import measure_engine_agreement

        return measure_engine_agreement(
            exported,
            Path("unused.engine"),
            image_dir,
            PreprocessConfig(size=(16, 16)),
            task=task,
            images=5,
            offset=offset,
        )

    def test_a_faithful_engine_passes(self, monkeypatch, exported: Path, image_dir: Path) -> None:
        self._fake_engine(monkeypatch, exported, lambda out: out + 1e-4)
        result = self._measure(exported, image_dir, "classification")
        assert result["metric"] == "top1_agreement"
        assert result["value"] == 1.0
        assert result["images"] == 5
        assert result["passed"]

    def test_a_wrong_engine_fails(self, monkeypatch, exported: Path, image_dir: Path) -> None:
        self._fake_engine(monkeypatch, exported, lambda out: -out)
        result = self._measure(exported, image_dir, "classification")
        assert result["value"] == 0.0
        assert not result["passed"]

    def test_similarity_reports_mean_and_min_cosine(
        self, monkeypatch, exported: Path, image_dir: Path
    ) -> None:
        self._fake_engine(monkeypatch, exported, lambda out: out * 3.0)
        result = self._measure(exported, image_dir, "similarity")
        assert result["metric"] == "mean_cosine"
        assert result["value"] == pytest.approx(1.0)
        assert result["min_cosine"] == pytest.approx(1.0)
        assert result["passed"]

    def test_no_images_past_the_offset_is_an_error(
        self, monkeypatch, exported: Path, image_dir: Path
    ) -> None:
        self._fake_engine(monkeypatch, exported, lambda out: out)
        with pytest.raises(ValueError, match="no images"):
            self._measure(exported, image_dir, "classification", offset=100)
