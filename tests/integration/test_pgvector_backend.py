"""The pgvector backend against a real Postgres.

Skips unless one is reachable. The SQL here cannot be checked any other way:
a vector column, the `<=>` cosine operator and an upsert are all things the
database either accepts or does not, and a mock would only assert that the
strings match what was written.

Start one with:

    docker run -d --rm --name pgvector-test -p 55432:5432 \\
      -e POSTGRES_USER=testuser -e POSTGRES_PASSWORD=testpass \\
      -e POSTGRES_DB=testdb pgvector/pgvector:pg16

    export PGVECTOR_TEST_URL=postgresql+asyncpg://testuser:testpass@localhost:55432/testdb
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration]

TEST_URL = os.getenv(
    "PGVECTOR_TEST_URL",
    "postgresql+asyncpg://testuser:testpass@localhost:55432/testdb",
)
DIMENSION = 8


async def _reachable(url: str) -> bool:
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(url, pool_pre_ping=True)
        async with engine.connect():
            pass
        await engine.dispose()
    except Exception:
        return False
    return True


@pytest.fixture
async def index(monkeypatch):
    """A PgVectorSimilarityIndex wired to the test database, starting empty."""
    if not await _reachable(TEST_URL):
        pytest.skip(f"no Postgres at {TEST_URL}")

    from api.services.db_service import DatabaseService
    from api.services.pgvector_index import PgVectorSimilarityIndex

    service = DatabaseService()
    service.settings = service.settings.model_copy(update={"database_url": TEST_URL})
    assert await service.connect(create_tables=False), "could not connect to the test database"

    monkeypatch.setattr("api.services.pgvector_index.get_db_service", lambda: service)

    backend = PgVectorSimilarityIndex(dimension=DIMENSION)
    assert await backend.ensure_schema(), "pgvector extension is missing from this Postgres"
    await backend.reset()
    try:
        yield backend
    finally:
        await backend.reset()
        await service.close()


def _unit(*values: float) -> np.ndarray:
    vec = np.zeros(DIMENSION, dtype=np.float32)
    for i, v in enumerate(values):
        vec[i] = v
    return vec


class TestRoundTrip:
    async def test_an_inserted_vector_is_found_again(self, index) -> None:
        item_id = await index.insert(_unit(1, 0, 0), label="dog", metadata={"src": "test"})

        hits = await index.query(_unit(1, 0, 0), top_k=5)
        assert len(hits) == 1

        item, score = hits[0]
        assert item.id == item_id
        assert item.label == "dog"
        assert item.metadata == {"src": "test"}
        assert score == pytest.approx(1.0, abs=1e-5), "identical vectors should score 1.0"

    async def test_results_come_back_most_similar_first(self, index) -> None:
        await index.insert(_unit(1, 0, 0), item_id="same")
        await index.insert(_unit(0.7, 0.7, 0), item_id="near")
        await index.insert(_unit(0, 1, 0), item_id="far")

        hits = await index.query(_unit(1, 0, 0), top_k=3)
        assert [item.id for item, _ in hits] == ["same", "near", "far"]
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    async def test_min_similarity_filters(self, index) -> None:
        await index.insert(_unit(1, 0, 0), item_id="same")
        await index.insert(_unit(0, 1, 0), item_id="orthogonal")

        hits = await index.query(_unit(1, 0, 0), top_k=10, min_similarity=0.5)
        assert [item.id for item, _ in hits] == ["same"]

    async def test_top_k_limits_the_result(self, index) -> None:
        for i in range(10):
            await index.insert(_unit(1, i / 10), item_id=f"v{i}")
        assert len(await index.query(_unit(1, 0), top_k=3)) == 3


class TestWrites:
    async def test_reindexing_the_same_id_replaces_it(self, index) -> None:
        """Two replicas racing on one id must converge, not duplicate."""
        await index.insert(_unit(1, 0, 0), item_id="fixed", label="first")
        await index.insert(_unit(0, 1, 0), item_id="fixed", label="second")

        assert await index.count() == 1
        hits = await index.query(_unit(0, 1, 0), top_k=5)
        assert hits[0][0].label == "second"
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

    async def test_delete_removes_the_row(self, index) -> None:
        item_id = await index.insert(_unit(1, 0, 0))
        assert await index.delete(item_id) is True
        assert await index.count() == 0
        assert await index.delete(item_id) is False

    async def test_reset_empties_the_index(self, index) -> None:
        for _ in range(3):
            await index.insert(_unit(1, 0, 0))
        await index.reset()
        assert await index.count() == 0

    async def test_reset_with_a_new_dimension_rebuilds_the_table(self, index) -> None:
        """The dimension is in the column type, so old rows cannot survive it."""
        await index.insert(_unit(1, 0, 0))
        await index.reset(dimension=16)

        assert index.dimension == 16
        assert await index.count() == 0

        wide = np.zeros(16, dtype=np.float32)
        wide[0] = 1.0
        assert await index.insert(wide)
        assert await index.count() == 1
        await index.reset(dimension=DIMENSION)

    async def test_a_vector_of_the_wrong_width_is_refused(self, index) -> None:
        with pytest.raises(ValueError, match="dimensional"):
            await index.insert(np.ones(DIMENSION + 1, dtype=np.float32))


class TestSharedAcrossInstances:
    """The whole point: two processes, one index."""

    async def test_a_second_instance_sees_the_first_one_s_writes(self, index) -> None:
        from api.services.pgvector_index import PgVectorSimilarityIndex

        await index.insert(_unit(1, 0, 0), item_id="written-by-a", label="from A")

        replica = PgVectorSimilarityIndex(dimension=DIMENSION)
        assert await replica.ensure_schema()

        hits = await replica.query(_unit(1, 0, 0), top_k=5)
        assert [item.id for item, _ in hits] == ["written-by-a"]
        assert await replica.count() == 1


class TestStats:
    async def test_snapshot_reports_a_shared_backend(self, index) -> None:
        await index.insert(_unit(1, 0, 0))
        snapshot = await index.snapshot()
        assert snapshot["backend"] == "pgvector"
        assert snapshot["shared"] is True
        assert snapshot["size"] == 1
        assert snapshot["dimension"] == DIMENSION
