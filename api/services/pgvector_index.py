"""Similarity index backed by Postgres and pgvector.

Plain English:
    The default index keeps every embedding in one process's memory. Run two
    API replicas and you get two separate indexes: an image indexed through
    replica A cannot be found through replica B, and a restart loses whatever
    was not saved to disk. This puts the vectors in Postgres instead, so every
    replica reads and writes one index.

Postgres was chosen over FAISS or a dedicated vector database because it is
already running in this stack. No new service, no new failure mode, no second
thing to back up. pgvector is an extension, not a fork, so `CREATE EXTENSION
vector` is the whole installation.

The tradeoff is latency. An in-memory dot product over 10,000 vectors takes
tens of microseconds; the same search here is a network round trip plus a
query, so hundreds of microseconds to low milliseconds. That is irrelevant
next to 15 ms of inference, and it buys correctness under scaling.

Indexing strategy: none, deliberately. pgvector's HNSW and IVFFlat indexes are
approximate, and the exact scan this does is fast to roughly a million rows.
Adding an approximate index before it is needed would trade recall for speed
nobody is asking for. The note in `snapshot()` says when to revisit.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import text

from api.config import Settings, settings
from api.logging_config import get_logger
from api.services.db_service import get_db_service
from api.services.similarity_index import IndexedItem

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


class PgVectorSimilarityIndex:
    """Exact cosine search over embeddings stored in Postgres.

    Implements the same async surface as :class:`SimilarityIndex`, so the
    router holds one or the other without knowing which.
    """

    def __init__(self, dimension: int | None = None, config: Settings | None = None) -> None:
        self.settings = config or settings
        self.dimension = int(dimension or self.settings.similarity_dimension)
        self._ready = False

    # ------------------------------------------------------------- setup --
    async def ensure_schema(self) -> bool:
        """Create the extension and table if they are missing.

        Returns False rather than raising when the database is unreachable or
        pgvector is not installed. Similarity is one feature of the API; it
        should degrade rather than stop the process from starting.
        """
        if self._ready:
            return True

        db = get_db_service()
        if not db.available:
            return False

        try:
            async with db.session() as session:
                await self._create_extension(session)
                await session.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS similarity_vectors ("
                        "  id         TEXT PRIMARY KEY,"
                        f"  embedding  vector({self.dimension}) NOT NULL,"
                        "  label      TEXT,"
                        "  metadata   JSONB,"
                        "  added_at   DOUBLE PRECISION NOT NULL"
                        ")"
                    )
                )
            self._ready = True
            return True
        except Exception as exc:
            # Most often: the image is plain postgres, without pgvector.
            logger.warning(
                "pgvector_schema_unavailable",
                extra={"error": f"{type(exc).__name__}: {exc}", "dimension": self.dimension},
            )
            return False

    @staticmethod
    async def _create_extension(session: Any) -> None:
        """Create the vector extension, tolerating a concurrent creator.

        `CREATE EXTENSION IF NOT EXISTS` is not atomic. Two replicas starting
        together both see it missing, both try, and the loser gets a unique
        violation on pg_extension_name_index. Observed on a two-replica
        rollout: one pod came up with similarity degraded for no reason it
        could report.

        A savepoint keeps that failure from poisoning the outer transaction,
        which would otherwise take the CREATE TABLE down with it.
        """
        from sqlalchemy.exc import DBAPIError, IntegrityError

        try:
            async with session.begin_nested():
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except (IntegrityError, DBAPIError) as exc:
            # Someone else won the race. Anything else is a real problem.
            if "pg_extension_name_index" not in str(exc) and "already exists" not in str(exc):
                raise
            logger.debug("vector extension was created concurrently by another replica")

    def _to_literal(self, vector: np.ndarray) -> str:
        """pgvector's text input format, which is '[1,2,3]'.

        Passed as a bound parameter and cast in SQL rather than interpolated,
        so this never becomes an injection site.
        """
        return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"

    def _normalise(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dimension:
            raise ValueError(f"expected a {self.dimension}-dimensional vector, got {vec.shape[0]}")
        # Normalising on write means cosine distance and inner product agree,
        # and a later switch to an inner-product index needs no migration.
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    # ------------------------------------------------------------- write --
    async def insert(
        self,
        vector: np.ndarray,
        *,
        item_id: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        import json
        import time

        vec = self._normalise(vector)
        if not await self.ensure_schema():
            raise RuntimeError("the similarity index is not available: pgvector schema missing")

        item_id = item_id or uuid.uuid4().hex
        db = get_db_service()
        async with db.session() as session:
            # Upsert, so re-indexing the same id replaces rather than
            # duplicating. Two replicas racing on one id then converge instead
            # of leaving the index with two rows that both claim it.
            await session.execute(
                text(
                    "INSERT INTO similarity_vectors (id, embedding, label, metadata, added_at) "
                    "VALUES (:id, CAST(:embedding AS vector), :label, "
                    "        CAST(:metadata AS JSONB), :added_at) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "  embedding = EXCLUDED.embedding,"
                    "  label     = EXCLUDED.label,"
                    "  metadata  = EXCLUDED.metadata,"
                    "  added_at  = EXCLUDED.added_at"
                ),
                {
                    "id": item_id,
                    "embedding": self._to_literal(vec),
                    "label": label,
                    "metadata": json.dumps(metadata) if metadata is not None else None,
                    "added_at": time.time(),
                },
            )
        return item_id

    async def delete(self, item_id: str) -> bool:
        if not await self.ensure_schema():
            return False
        db = get_db_service()
        async with db.session() as session:
            result = await session.execute(
                text("DELETE FROM similarity_vectors WHERE id = :id"), {"id": item_id}
            )
            # execute() is typed as Result, which has no rowcount; a DELETE
            # returns a CursorResult, which does.
            return bool(getattr(result, "rowcount", 0))

    async def reset(self, dimension: int | None = None) -> None:
        """Empty the index, and change its dimension if asked.

        A dimension change drops the table rather than truncating it: the
        column type carries the dimension, so rows of the old width cannot
        coexist with the new ones.
        """
        db = get_db_service()
        if not db.available:
            return

        if dimension is not None and int(dimension) != self.dimension:
            self.dimension = int(dimension)
            self._ready = False
            async with db.session() as session:
                await session.execute(text("DROP TABLE IF EXISTS similarity_vectors"))
            await self.ensure_schema()
            return

        if await self.ensure_schema():
            async with db.session() as session:
                await session.execute(text("TRUNCATE similarity_vectors"))

    # -------------------------------------------------------------- read --
    async def query(
        self, vector: np.ndarray, top_k: int = 10, min_similarity: float = -1.0
    ) -> list[tuple[IndexedItem, float]]:
        """Nearest neighbours by cosine similarity, closest first."""
        if not await self.ensure_schema():
            return []
        vec = self._normalise(vector)

        # `<=>` is cosine DISTANCE, so similarity is 1 - distance. Ordering by
        # the operator (not by the derived similarity) is what lets pgvector
        # use an index later without the query changing.
        db = get_db_service()
        async with db.session() as session:
            rows: Sequence[Any] = (
                await session.execute(
                    text(
                        "SELECT id, label, metadata, added_at, "
                        "       1 - (embedding <=> CAST(:q AS vector)) AS similarity "
                        "FROM similarity_vectors "
                        "ORDER BY embedding <=> CAST(:q AS vector) "
                        "LIMIT :k"
                    ),
                    {"q": self._to_literal(vec), "k": max(1, int(top_k))},
                )
            ).all()

        hits: list[tuple[IndexedItem, float]] = []
        for row in rows:
            score = float(row.similarity)
            if score < min_similarity:
                continue
            hits.append(
                (
                    IndexedItem(
                        id=row.id,
                        label=row.label,
                        metadata=row.metadata,
                        added_at=float(row.added_at),
                    ),
                    score,
                )
            )
        return hits

    async def count(self) -> int:
        if not await self.ensure_schema():
            return 0
        db = get_db_service()
        async with db.session() as session:
            return int(
                (
                    await session.execute(text("SELECT COUNT(*) FROM similarity_vectors"))
                ).scalar_one()
            )

    async def snapshot(self) -> dict[str, Any]:
        size = await self.count()
        return {
            "size": size,
            "dimension": self.dimension,
            "backend": "pgvector",
            "shared": True,
            "search": "exact",
            # Says when the default stops being the right default, so the
            # decision is visible to whoever is looking at the number.
            "note": (
                (
                    "exact scan, linear in index size. Fine to roughly 1M rows; "
                    "past that add an HNSW index and accept approximate recall."
                )
                if size > 100_000
                else None
            ),
        }

    # The in-memory backend persists to disk. This one is already durable, so
    # these exist only so the two are interchangeable.
    def save(self, path: Any = None) -> None:
        return None

    def load(self, path: Any = None) -> bool:
        return True


__all__ = ["PgVectorSimilarityIndex"]
