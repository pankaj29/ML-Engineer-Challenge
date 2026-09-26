"""Vector index for image similarity search.

Plain English:
    An embedding model turns each image into a list of numbers (a "vector")
    positioned so that similar-looking images land near each other. Searching
    then means: embed the query image, and find the stored vectors closest
    to it.

    "Closest" here is **cosine similarity** — how closely two vectors point in
    the same direction, ignoring their length. Because we store every vector
    normalised to length 1, the cosine similarity is just the dot product, so
    searching the whole index is a single matrix multiplication.

Scale, honestly stated:
    This is an exact, brute-force search. It compares the query against every
    stored vector. On a matrix of 100,000 x 512 floats that is roughly 50 M
    multiply-adds — a few milliseconds in NumPy, which is entirely fine.
    Beyond roughly a million vectors you want an approximate index (FAISS
    HNSW, pgvector, a dedicated vector database), which trades a little recall
    for a large speed-up. The interface here is deliberately narrow so that
    swap is a contained change.

Persistence is a plain ``.npz`` file plus a JSON sidecar, so the index
survives a restart without needing another service.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from api.config import Settings, settings
from api.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IndexedItem:
    """Metadata stored alongside one vector."""

    id: str
    label: str | None = None
    metadata: dict[str, Any] | None = None
    added_at: float = field(default_factory=time.time)


class SimilarityIndex:
    """Exact cosine-similarity search over a set of image embeddings.

    Thread-safe: the API may add and search concurrently, and NumPy arrays
    are replaced wholesale on insert rather than mutated, so a search never
    observes a half-written array.
    """

    def __init__(self, dimension: int = 512, config: Settings | None = None) -> None:
        self.settings = config or settings
        self.dimension = dimension
        # Shape (N, dimension). Every row is L2-normalised on insert.
        self._vectors: np.ndarray = np.zeros((0, dimension), dtype=np.float32)
        self._items: list[IndexedItem] = []
        self._lock = threading.RLock()
        self._path = Path(self.settings.model_artifacts_dir) / "similarity_index.npz"
        self._meta_path = Path(self.settings.model_artifacts_dir) / "similarity_index.json"

    def __len__(self) -> int:
        return len(self._items)

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        return len(self._items)

    # --------------------------------------------------------------- write --
    def add(
        self,
        vector: np.ndarray,
        *,
        item_id: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add one vector to the index and return its id.

        The vector is normalised here rather than trusting the caller, because
        a single un-normalised row would silently corrupt every similarity
        score computed against it.

        Raises:
            ValueError: The vector has the wrong dimensionality.
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dimension:
            raise ValueError(f"expected a {self.dimension}-dimensional vector, got {vec.shape[0]}")

        norm = float(np.linalg.norm(vec))
        vec = vec / max(norm, 1e-12)

        item_id = item_id or uuid.uuid4().hex
        with self._lock:
            self._vectors = np.vstack([self._vectors, vec[None, :]])
            self._items.append(IndexedItem(id=item_id, label=label, metadata=metadata))
        return item_id

    def add_many(self, vectors: np.ndarray, items: list[IndexedItem] | None = None) -> list[str]:
        """Add several vectors at once.

        Much faster than repeated :meth:`add` calls, which each re-allocate
        and copy the whole matrix.
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise ValueError(f"expected shape (N, {self.dimension}), got {arr.shape}")

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)

        items = items or [IndexedItem(id=uuid.uuid4().hex) for _ in range(arr.shape[0])]
        with self._lock:
            self._vectors = np.vstack([self._vectors, arr])
            self._items.extend(items)
        return [i.id for i in items]

    def remove(self, item_id: str) -> bool:
        """Delete one vector by id. Returns False if it was not present."""
        with self._lock:
            for idx, item in enumerate(self._items):
                if item.id == item_id:
                    self._vectors = np.delete(self._vectors, idx, axis=0)
                    self._items.pop(idx)
                    return True
        return False

    def clear(self) -> None:
        """Empty the index."""
        with self._lock:
            self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
            self._items.clear()

    # ---------------------------------------------------------------- read --
    def search(
        self, query: np.ndarray, top_k: int = 10, min_similarity: float = -1.0
    ) -> list[tuple[IndexedItem, float]]:
        """Find the vectors most similar to ``query``.

        Args:
            query: The query embedding. Normalised here if it is not already.
            top_k: How many neighbours to return.
            min_similarity: Drop results scoring below this. Cosine similarity
                runs from -1 (opposite) through 0 (unrelated) to 1 (identical).

        Returns:
            ``(item, score)`` pairs, most similar first.
        """
        with self._lock:
            if self._vectors.shape[0] == 0:
                return []
            vectors = self._vectors
            items = list(self._items)

        vec = np.asarray(query, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dimension:
            raise ValueError(f"query has {vec.shape[0]} dimensions, index has {self.dimension}")
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)

        # Both sides are unit length, so the dot product is the cosine.
        scores = vectors @ vec

        k = min(top_k, scores.shape[0])
        # argpartition avoids sorting all N scores when we only need the top k.
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            (items[int(i)], float(scores[int(i)]))
            for i in top_idx
            if float(scores[int(i)]) >= min_similarity
        ]

    # ------------------------------------------------------------ persist --
    def save(self, path: Path | None = None) -> Path:
        """Write the index to disk."""
        target = Path(path) if path else self._path
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            np.savez_compressed(target, vectors=self._vectors)
            meta = {
                "dimension": self.dimension,
                "count": len(self._items),
                "items": [
                    {
                        "id": i.id,
                        "label": i.label,
                        "metadata": i.metadata,
                        "added_at": i.added_at,
                    }
                    for i in self._items
                ],
            }
        meta_target = target.with_suffix(".json")
        meta_target.write_text(json.dumps(meta), encoding="utf-8")
        logger.info("similarity_index_saved", extra={"path": str(target), "count": meta["count"]})
        return target

    def load(self, path: Path | None = None) -> bool:
        """Load the index from disk. Returns False when there is nothing to load."""
        target = Path(path) if path else self._path
        meta_target = target.with_suffix(".json")
        if not target.exists() or not meta_target.exists():
            return False
        try:
            data = np.load(target)
            meta = json.loads(meta_target.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("similarity_index_load_failed", extra={"error": str(exc)})
            return False

        with self._lock:
            self.dimension = int(meta.get("dimension", self.dimension))
            self._vectors = data["vectors"].astype(np.float32)
            self._items = [
                IndexedItem(
                    id=i["id"],
                    label=i.get("label"),
                    metadata=i.get("metadata"),
                    added_at=i.get("added_at", time.time()),
                )
                for i in meta.get("items", [])
            ]

        # A mismatch means the two files are from different saves; refusing to
        # serve is better than returning results attributed to the wrong image.
        if self._vectors.shape[0] != len(self._items):
            logger.error(
                "similarity_index_inconsistent",
                extra={"vectors": self._vectors.shape[0], "items": len(self._items)},
            )
            self.clear()
            return False

        logger.info("similarity_index_loaded", extra={"count": len(self._items)})
        return True

    # ----------------------------------------------------------- async --
    # The router awaits these so it does not care which backend it holds.
    # pgvector has to go to the database for all of them; this one does not,
    # so they return immediately without yielding to the event loop.

    async def insert(
        self,
        vector: np.ndarray,
        *,
        item_id: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.add(vector, item_id=item_id, label=label, metadata=metadata)

    async def query(
        self, vector: np.ndarray, top_k: int = 10, min_similarity: float = -1.0
    ) -> list[tuple[IndexedItem, float]]:
        return self.search(vector, top_k=top_k, min_similarity=min_similarity)

    async def delete(self, item_id: str) -> bool:
        return self.remove(item_id)

    async def reset(self, dimension: int | None = None) -> None:
        if dimension is not None:
            self.dimension = int(dimension)
            self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
            self._items.clear()
            return
        self.clear()

    async def count(self) -> int:
        return self.size

    async def snapshot(self) -> dict[str, Any]:
        return self.stats()

    # ------------------------------------------------------------ report --
    def stats(self) -> dict[str, Any]:
        """Index statistics, surfaced through the health endpoint."""
        with self._lock:
            return {
                "size": len(self._items),
                "dimension": self.dimension,
                "memory_mb": round(self._vectors.nbytes / 1_048_576, 2),
                "persisted": self._path.exists(),
                "backend": "memory",
                "shared": False,
            }


# Process-wide singleton, created in the application lifespan handler.
# Either backend; both expose the same async interface.
_index: Any = None


def get_similarity_index() -> Any:
    """FastAPI dependency returning whichever index backend is configured.

    ``similarity_backend="pgvector"`` gives one index shared by every replica.
    The default keeps vectors in this process, which is faster and needs no
    database, but means N replicas hold N unrelated indexes.
    """
    global _index
    if _index is None:
        if settings.similarity_backend == "pgvector":
            from api.services.pgvector_index import PgVectorSimilarityIndex

            _index = PgVectorSimilarityIndex()
        else:
            _index = SimilarityIndex()
            _index.load()
    return _index


def set_similarity_index(index: Any | None) -> None:
    """Replace the singleton. Used by the lifespan handler and by tests."""
    global _index
    _index = index


__all__ = [
    "IndexedItem",
    "SimilarityIndex",
    "get_similarity_index",
    "set_similarity_index",
]
