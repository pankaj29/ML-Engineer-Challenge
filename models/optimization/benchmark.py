"""Inference benchmarking across model formats.

Plain English:
    This measures how fast each version of a model actually runs, so the claim
    "ONNX is faster" is a number in a report rather than folklore.

Three things make a benchmark trustworthy, and all three are enforced here:

* **Warmup.** The first few runs of any runtime are unrepresentatively slow —
  memory arenas are allocated, kernels are selected, caches are cold.
  Discarding them is the difference between a real number and a scary one.
* **Percentiles, not averages.** An average hides the tail. If 1 in 20
  requests takes 3 seconds, users notice, but the mean barely moves. We report
  p50, p95 and p99, because a latency SLA is written against p95/p99.
* **Enough samples.** A handful of runs on a busy laptop measures the laptop,
  not the model. The default of 100 iterations keeps the percentiles stable.

Usage::

    python -m models.optimization.benchmark --artifacts models/artifacts --iterations 100
    python -m models.optimization.benchmark --onnx models/artifacts/resnet18.onnx --batch-sizes 1,4,8
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BenchmarkResult:
    """Latency statistics for one model/format/batch-size combination."""

    name: str
    runtime: str
    device: str
    batch_size: int
    iterations: int
    mean_ms: float
    median_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    throughput_ips: float  # images per second
    size_mb: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def per_image_ms(self) -> float:
        """Latency divided by batch size — the fair cross-batch comparison."""
        return self.mean_ms / self.batch_size if self.batch_size else self.mean_ms

    def summary(self) -> str:
        return (
            f"{self.name:<28} {self.runtime:<12} {self.device:<5} "
            f"b={self.batch_size:<3} "
            f"p50={self.p50_ms:7.2f}ms  p95={self.p95_ms:7.2f}ms  "
            f"{self.throughput_ips:7.1f} img/s  {self.size_mb:6.1f}MB"
        )


def measure(
    fn: Callable[[], Any],
    *,
    iterations: int = 100,
    warmup: int = 10,
) -> list[float]:
    """Time a callable repeatedly and return per-call durations in milliseconds.

    ``time.perf_counter`` is used rather than ``time.time`` because it is
    monotonic and has nanosecond resolution; ``time.time`` can jump backwards
    when the system clock is adjusted, producing negative durations.
    """
    for _ in range(warmup):
        fn()

    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def summarise(
    timings: list[float],
    *,
    name: str,
    runtime: str,
    device: str,
    batch_size: int,
    size_mb: float = 0.0,
    notes: list[str] | None = None,
) -> BenchmarkResult:
    """Turn a list of durations into a :class:`BenchmarkResult`."""
    ordered = sorted(timings)
    n = len(ordered)

    def pct(p: float) -> float:
        """Nearest-rank percentile; stable for small sample counts."""
        if n == 0:
            return 0.0
        idx = min(n - 1, max(0, int(round(p / 100.0 * n)) - 1))
        return ordered[idx]

    mean = statistics.fmean(ordered) if ordered else 0.0
    return BenchmarkResult(
        name=name,
        runtime=runtime,
        device=device,
        batch_size=batch_size,
        iterations=n,
        mean_ms=mean,
        median_ms=statistics.median(ordered) if ordered else 0.0,
        p50_ms=pct(50),
        p90_ms=pct(90),
        p95_ms=pct(95),
        p99_ms=pct(99),
        min_ms=ordered[0] if ordered else 0.0,
        max_ms=ordered[-1] if ordered else 0.0,
        stdev_ms=statistics.stdev(ordered) if n > 1 else 0.0,
        throughput_ips=(batch_size * 1000.0 / mean) if mean else 0.0,
        size_mb=size_mb,
        notes=notes or [],
    )


def benchmark_onnx(
    path: Path,
    *,
    input_shape: tuple[int, ...] = (3, 224, 224),
    batch_sizes: tuple[int, ...] = (1,),
    iterations: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    """Benchmark an ONNX model across the given batch sizes."""
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), opts, providers=providers)

    actual_device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"

    # ONNX Runtime falls back to CPU without raising when the CUDA provider is
    # missing. Benchmarks that silently measure the wrong device are worse
    # than no benchmarks: they get written into a report, compared against
    # other runtimes, and quoted. Say it plainly.
    if device == "cuda" and actual_device == "cpu":
        available = ort.get_available_providers()
        warnings.warn(
            "CUDA was requested but ONNX Runtime has no CUDAExecutionProvider, "
            "so these numbers are CPU numbers. Available providers: "
            f"{available}. Install onnxruntime-gpu matching this CUDA version, "
            "and make sure plain onnxruntime is not also installed - whichever "
            "imports first wins.",
            RuntimeWarning,
            stacklevel=2,
        )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    size_mb = path.stat().st_size / 1_048_576

    results: list[BenchmarkResult] = []
    for batch in batch_sizes:
        data = np.random.randn(batch, *input_shape).astype(np.float32)
        try:
            # `data=data` binds the current iteration's array into the
            # lambda. Without it the closure would read whatever `data` holds
            # when it is finally called - safe here, but the kind of latent
            # bug that appears the moment execution becomes deferred.
            timings = measure(
                lambda data=data: session.run(output_names, {input_name: data}),
                iterations=iterations,
                warmup=warmup,
            )
        except Exception as exc:
            results.append(
                summarise(
                    [],
                    name=path.stem,
                    runtime="onnx",
                    device=actual_device,
                    batch_size=batch,
                    size_mb=size_mb,
                    notes=[f"failed at batch {batch}: {type(exc).__name__}: {exc}"],
                )
            )
            continue
        results.append(
            summarise(
                timings,
                name=path.stem,
                runtime="onnx_int8" if "int8" in path.stem else "onnx",
                device=actual_device,
                batch_size=batch,
                size_mb=size_mb,
            )
        )
    return results


def benchmark_interleaved(
    cases: list[tuple[Path, tuple[int, ...]]],
    *,
    batch_sizes: tuple[int, ...] = (1,),
    iterations: int = 100,
    warmup: int = 10,
    rounds: int = 5,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    """Benchmark several models in rotation rather than one after another.

    On a laptop CPU, run-to-run conditions drift: thermal limits, power
    states, and on a hybrid part the scheduler moving threads between
    performance and efficiency cores. Timing each model in one block hands
    that drift to whichever model happens to run during a slow phase. Two
    ResNet-50s that differ only in their final layer once measured 36 and
    61 ms p50 in the same run that way. Here every (model, batch) case gets
    ``iterations / rounds`` timed calls per round, round after round, so each
    one samples the same spread of machine conditions.
    """
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    rounds = max(1, rounds)
    per_round = max(1, iterations // rounds)

    runs: list[dict[str, Any]] = []
    for path, input_shape in cases:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), opts, providers=providers)
        actual = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        for batch in batch_sizes:
            data = np.random.randn(batch, *input_shape).astype(np.float32)
            runs.append(
                {
                    "path": path,
                    "batch": batch,
                    "device": actual,
                    "fn": lambda s=session, o=output_names, n=input_name, d=data: s.run(o, {n: d}),
                    "timings": [],
                    "error": None,
                }
            )

    for run in runs:
        try:
            for _ in range(warmup):
                run["fn"]()
        except Exception as exc:
            run["error"] = f"failed at batch {run['batch']}: {type(exc).__name__}: {exc}"

    for _ in range(rounds):
        for run in runs:
            if run["error"] is None:
                run["timings"].extend(measure(run["fn"], iterations=per_round, warmup=0))

    results: list[BenchmarkResult] = []
    for run in runs:
        path = run["path"]
        notes = [run["error"]] if run["error"] else []
        if device == "cuda" and run["device"] == "cpu":
            notes.append("CUDA requested but ONNX Runtime ran on CPU: these are CPU numbers")
        results.append(
            summarise(
                run["timings"],
                name=path.stem,
                runtime="onnx_int8" if "int8" in path.stem else "onnx",
                device=run["device"],
                batch_size=run["batch"],
                size_mb=path.stat().st_size / 1_048_576,
                notes=notes,
            )
        )
    return results


def benchmark_torch(
    model: Any,
    *,
    name: str = "model",
    input_shape: tuple[int, ...] = (3, 224, 224),
    batch_sizes: tuple[int, ...] = (1,),
    iterations: int = 100,
    warmup: int = 10,
    device: str = "cpu",
    size_mb: float = 0.0,
) -> list[BenchmarkResult]:
    """Benchmark an eager or TorchScript PyTorch model."""
    import torch

    model = model.eval().to(device)
    results: list[BenchmarkResult] = []

    for batch in batch_sizes:
        data = torch.randn(batch, *input_shape, device=device)

        def run(data: Any = data) -> None:
            with torch.inference_mode():
                model(data)
            # A CUDA call is asynchronous: without synchronising we would be
            # timing how fast Python can queue work, not how fast it runs.
            if device == "cuda":
                torch.cuda.synchronize()

        timings = measure(run, iterations=iterations, warmup=warmup)
        results.append(
            summarise(
                timings,
                name=name,
                runtime="torch",
                device=device,
                batch_size=batch,
                size_mb=size_mb,
            )
        )
    return results


def environment_info() -> dict[str, Any]:
    """Record the machine the benchmark ran on.

    Latency numbers are meaningless without this: 30 ms on a laptop and 30 ms
    on a 64-core server say completely different things.
    """
    info: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": None,
        "gpu": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    return info


def render_markdown(results: list[BenchmarkResult], env: dict[str, Any]) -> str:
    """Render results as a Markdown report for ``benchmarks/reports/``."""
    lines = [
        "# Inference Benchmark Report",
        "",
        f"Generated: {env.get('timestamp', 'unknown')}",
        "",
        "## Environment",
        "",
        "| Property | Value |",
        "| --- | --- |",
    ]
    for key in (
        "platform",
        "processor",
        "cpu_count",
        "python",
        "torch",
        "onnxruntime",
        "gpu",
        "cuda",
    ):
        if env.get(key) is not None:
            lines.append(f"| {key} | {env[key]} |")

    lines += [
        "",
        "## Results",
        "",
        "Latency is wall-clock time for one forward pass. `per-image` divides by",
        "batch size, which is the fair way to compare across batch sizes.",
        "",
        "| Model | Runtime | Device | Batch | p50 (ms) | p95 (ms) | p99 (ms) | per-image (ms) | Throughput (img/s) | Size (MB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda x: (x.name, x.batch_size)):
        lines.append(
            f"| {r.name} | {r.runtime} | {r.device} | {r.batch_size} | "
            f"{r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} | "
            f"{r.per_image_ms:.2f} | {r.throughput_ips:.1f} | {r.size_mb:.1f} |"
        )

    # Speed-up table against the float32 ONNX baseline at batch 1. Quantised
    # artifacts carry a variant after "_int8" (resnet50_int8_static,
    # ..._int8_trt), so the whole suffix goes, not just "_int8".
    def base_name(name: str) -> str:
        return re.sub(r"_int8(_\w+)?$", "", name)

    baselines = {
        r.name: r
        for r in results
        if r.batch_size == 1 and r.runtime == "onnx" and "int8" not in r.name
    }
    comparisons = [
        (r, baselines[base_name(r.name)])
        for r in results
        if r.batch_size == 1 and base_name(r.name) in baselines
    ]
    if comparisons:
        lines += [
            "",
            "## Speed-up vs float32 ONNX (batch 1)",
            "",
            "| Model | Runtime | p50 speed-up | Size reduction |",
            "| --- | --- | ---: | ---: |",
        ]
        for r, base in comparisons:
            if r is base:
                continue
            speedup = base.p50_ms / r.p50_ms if r.p50_ms else 0.0
            shrink = base.size_mb / r.size_mb if r.size_mb else 0.0
            lines.append(f"| {r.name} | {r.runtime} | {speedup:.2f}x | {shrink:.2f}x |")

    notes = [(r.name, n) for r in results for n in r.notes]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- **{name}**: {note}" for name, note in notes]

    lines += [
        "",
        "## How to read this",
        "",
        "- **p50** is the typical request. **p95/p99** are the slow tail users complain about.",
        "- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.",
        "- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether",
        "  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.",
        "",
    ]
    return "\n".join(lines)


def _registry_input_shapes() -> dict[str, tuple[int, ...]]:
    """Map each registered model's artifact stem to its input shape (C, H, W).

    Falls back to an empty mapping when the registry is unavailable, so the
    benchmark still runs on loose .onnx files.
    """
    try:
        from api.services.model_service import ModelService

        shapes: dict[str, tuple[int, ...]] = {}
        for entry in ModelService().list_entries():
            if not entry.input_shape:
                continue
            spatial = tuple(entry.input_shape[1:])  # drop the batch dimension
            for rel in entry.artifacts.values():
                shapes[Path(rel).stem.replace("_int8", "")] = spatial
        return shapes
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark model inference across formats.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO_ROOT / "models" / "artifacts",
        help="Directory to scan for .onnx files.",
    )
    parser.add_argument(
        "--onnx", type=Path, action="append", default=None, help="Benchmark a specific .onnx file."
    )
    parser.add_argument("--batch-sizes", default="1,4,8", help="Comma-separated batch sizes.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Interleave models over this many rounds; see benchmark_interleaved.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    batch_sizes = tuple(int(b) for b in args.batch_sizes.split(",") if b.strip())
    shape = (3, args.image_size, args.image_size)

    targets = args.onnx or sorted(args.artifacts.glob("*.onnx"))
    if not targets:
        print(f"error: no .onnx files found in {args.artifacts}", file=sys.stderr)
        return 2

    # A glob can only yield files that exist, but an explicit --onnx cannot.
    # Without this, a mistyped path surfaced as a raw ONNX Runtime NO_SUCHFILE
    # traceback partway through the run, after earlier models had already been
    # benchmarked and their results thrown away.
    missing = [p for p in targets if not Path(p).is_file()]
    if missing:
        for path in missing:
            print(f"error: no such model file: {path}", file=sys.stderr)
        return 2

    # Each model has its own input size (224 for the classifiers, 640 for the
    # detector). Benchmarking them all at one size would either crash the
    # detector or silently measure it at the wrong resolution, which would
    # make the comparison meaningless.
    shapes_by_stem = _registry_input_shapes()

    cases = [(path, shapes_by_stem.get(path.stem.replace("_int8", ""), shape)) for path in targets]
    print(
        f"benchmarking {len(cases)} models x {len(batch_sizes)} batch sizes, "
        f"{args.iterations} iterations in {args.rounds} interleaved rounds ..."
    )
    results = benchmark_interleaved(
        cases,
        batch_sizes=batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        rounds=args.rounds,
        device=device,
    )

    print()
    for r in sorted(results, key=lambda x: (x.name, x.batch_size)):
        print(r.summary())

    env = environment_info()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "benchmark_results.json"
    json_path.write_text(
        json.dumps({"environment": env, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )

    md_path = args.output_dir / "BENCHMARKS.md"
    md_path.write_text(render_markdown(results, env), encoding="utf-8")

    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
