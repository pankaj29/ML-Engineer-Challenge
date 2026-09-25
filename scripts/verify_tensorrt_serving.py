"""Prove the API serves predictions from a TensorRT engine.

Plain English:
    The benchmarks in `export_tensorrt.py` time the engine on its own. This
    answers a different question: does a request that arrives at the API get
    answered by the engine, or by something else?

The distinction is not pedantic. `TensorRTBackend`, the registry's `tensorrt`
artifact type and `PREFERRED_RUNTIME=tensorrt` all existed for a long time
without ever being exercised end to end, because the machines the test suite
runs on have no GPU.

**This must run as its own process.** `api.config` builds its settings
singleton when it is first imported, so setting `PREFERRED_RUNTIME` after any
`api.*` import has no effect: the app keeps the value from startup and serves
from ONNX while reporting success. That is why the environment is set at the
top of this file, above the imports, and why the notebook runs this with
`!python` rather than inline.

Usage::

    python scripts/verify_tensorrt_serving.py \\
        --model resnet50-tiny-imagenet --version 1.1.0-trt --image samples/dog.jpg
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set before any api.* import. setdefault, not assignment, so a caller can
# override any of them from the outside.
os.environ.setdefault("PREFERRED_RUNTIME", "tensorrt")
os.environ.setdefault("DEVICE", "cuda")
os.environ.setdefault("EAGER_MODEL_LOAD", "false")
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("ENVIRONMENT", "local")

import argparse
import json
import statistics
import time
from typing import Any


def _classify(client: Any, image: Path, model: str, version: str) -> Any:
    with image.open("rb") as handle:
        return client.post(
            "/api/v1/classify/upload",
            files={"file": (image.name, handle, "image/jpeg")},
            data={"model_name": model, "model_version": version},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="resnet50-tiny-imagenet")
    parser.add_argument("--version", default="1.1.0-trt")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "reports" / "tensorrt_serving.json",
    )
    args = parser.parse_args()

    image = args.image
    if image is None:
        candidates = sorted((REPO_ROOT / "samples").glob("*.jpg"))
        if not candidates:
            print("error: no image given and samples/ is empty", file=sys.stderr)
            return 2
        image = candidates[0]

    # Confirm the settings actually took, before drawing conclusions from a
    # run that may have been configured too late.
    from api.config import settings

    print(f"preferred_runtime : {settings.preferred_runtime}")
    print(f"device            : {settings.device}")
    if settings.preferred_runtime != "tensorrt":
        print(
            "\nerror: PREFERRED_RUNTIME did not take effect. Something imported "
            "api.config before this script set it, which means the result below "
            "would not be about TensorRT at all.",
            file=sys.stderr,
        )
        return 2

    from fastapi.testclient import TestClient

    from api.main import create_app

    with TestClient(create_app()) as client:
        response = _classify(client, image, args.model, args.version)
        if response.status_code != 200:
            print(f"\nrequest failed: {response.status_code}", file=sys.stderr)
            print(json.dumps(response.json(), indent=2)[:1500], file=sys.stderr)
            return 1

        body = response.json()
        runtime = body["model"]["runtime"]
        print(f"\nmodel   : {body['model']['name']} {body['model']['version']}")
        print(f"runtime : {runtime}")
        print(
            f"top     : {body['predictions'][0]['label']} "
            f"{body['predictions'][0]['confidence']:.3f}"
        )

        if runtime != "tensorrt":
            print(
                f"\nNOT TENSORRT: served by {runtime!r}, so this proves nothing.\n"
                "Register the engine as a version carrying only a tensorrt "
                "artifact; anything else gives the runtime chain something to "
                "fall back to.",
                file=sys.stderr,
            )
            return 1

        print("\nSERVED BY TENSORRT")

        for _ in range(args.warmup):
            _classify(client, image, args.model, args.version)

        latencies: list[float] = []
        reported: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            r = _classify(client, image, args.model, args.version)
            if r.status_code == 200:
                latencies.append((time.perf_counter() - started) * 1000)
                reported.append(r.json()["timing"]["inference_ms"])

        if not latencies:
            print("no successful calls during the benchmark", file=sys.stderr)
            return 1

        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)]
        model_p50 = statistics.median(reported)

        print(f"\ncalls             : {len(latencies)}")
        print(f"end-to-end p50    : {p50:.2f} ms")
        print(f"end-to-end p95    : {p95:.2f} ms")
        print(f"model inference   : {model_p50:.3f} ms (p50, as the API reports it)")
        print(f"overhead          : {p50 - model_p50:.2f} ms")
        print(
            "\nThe overhead is decode, preprocessing, middleware and "
            "serialisation. At this inference time it is most of the latency, "
            "which is where the next optimisation belongs."
        )

        record = {
            "model": f"{args.model}:{args.version}",
            "runtime": runtime,
            "image": image.name,
            "calls": len(latencies),
            "end_to_end_p50_ms": round(p50, 2),
            "end_to_end_p95_ms": round(p95, 2),
            "model_inference_p50_ms": round(model_p50, 3),
            "overhead_p50_ms": round(p50 - model_p50, 2),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"\nreport: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
