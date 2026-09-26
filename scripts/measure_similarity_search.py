"""Time exact similarity search at several index sizes.

Uses the in-process SimilarityIndex the API serves by default, filled with
random unit vectors of the embedding model's width (2048). Search cost does
not depend on what the vectors contain, only on how many there are.

    python scripts/measure_similarity_search.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.services.similarity_index import SimilarityIndex

DIM = 2048
SIZES = (1_000, 10_000, 100_000)
QUERIES = 50


def _run(size: int, rng: np.random.Generator) -> dict[str, float]:
    index = SimilarityIndex(dimension=DIM)
    vectors = rng.standard_normal((size, DIM)).astype(np.float32)
    index.add_many(vectors)
    query = vectors[0]
    timings = []
    for _ in range(QUERIES):
        started = time.perf_counter()
        index.search(query, top_k=10)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "vectors": size,
        "p50_ms": round(statistics.median(timings), 2),
        "memory_mb": round(size * DIM * 4 / 1_048_576, 1),
    }


def main() -> int:
    rng = np.random.default_rng(0)
    results = [_run(size, rng) for size in SIZES]
    for r in results:
        print(f"{r['vectors']:>8} vectors  p50 {r['p50_ms']:>7} ms  {r['memory_mb']:>7} MB")
    out = REPO_ROOT / "benchmarks" / "reports" / "similarity_search.json"
    out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "dimension": DIM,
                "queries": QUERIES,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
