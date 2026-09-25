"""The pgvector similarity backend.

No Postgres here. These cover the parts that are this module's own logic and
that would be wrong in a way a database round trip would not reveal: vector
normalisation, the literal format pgvector parses, and what happens when the
database is not there.

The query itself is exercised against a real Postgres in
tests/integration/test_pgvector_backend.py, which skips when none is running.
"""

from __future__ import annotations

import numpy as np
import pytest

from api.services.pgvector_index import PgVectorSimilarityIndex


@pytest.fixture
def index() -> PgVectorSimilarityIndex:
    return PgVectorSimilarityIndex(dimension=4)


class TestVectorEncoding:
    def test_vectors_are_normalised_on_write(self, index) -> None:
        """Cosine distance is only the cosine if both sides are unit length.

        One un-normalised row would score wrongly against every query without
        producing an error anywhere.
        """
        out = index._normalise(np.array([3.0, 4.0, 0.0, 0.0]))
        assert float(np.linalg.norm(out)) == pytest.approx(1.0)

    def test_a_zero_vector_does_not_divide_by_zero(self, index) -> None:
        out = index._normalise(np.zeros(4))
        assert np.isfinite(out).all()

    def test_the_wrong_dimension_is_refused(self, index) -> None:
        with pytest.raises(ValueError, match="4-dimensional"):
            index._normalise(np.ones(7))

    def test_the_literal_is_the_format_pgvector_parses(self, index) -> None:
        literal = index._to_literal(np.array([1.0, -0.5, 0.25, 0.0], dtype=np.float32))
        assert literal.startswith("[") and literal.endswith("]")
        assert literal.count(",") == 3
        assert " " not in literal

    def test_the_literal_survives_a_round_trip(self, index) -> None:
        original = index._normalise(np.array([0.1, 0.2, 0.3, 0.4]))
        parsed = np.array([float(x) for x in index._to_literal(original)[1:-1].split(",")])
        assert parsed == pytest.approx(original, abs=1e-6)


class TestWithoutADatabase:
    """Similarity is one feature. Losing Postgres must not take the API down."""

    @pytest.fixture(autouse=True)
    def _no_database(self, monkeypatch) -> None:
        class Unavailable:
            available = False

            def session(self):  # pragma: no cover - never reached
                raise AssertionError("session() called while the database was unavailable")

        monkeypatch.setattr("api.services.pgvector_index.get_db_service", lambda: Unavailable())

    async def test_schema_setup_reports_failure_instead_of_raising(self, index) -> None:
        assert await index.ensure_schema() is False

    async def test_search_returns_nothing(self, index) -> None:
        assert await index.query(np.ones(4)) == []

    async def test_count_is_zero(self, index) -> None:
        assert await index.count() == 0

    async def test_delete_reports_that_it_did_not(self, index) -> None:
        assert await index.delete("anything") is False

    async def test_insert_raises_because_it_cannot_silently_drop_data(self, index) -> None:
        """Reads degrade quietly; a write must not.

        Returning an id for a vector that was never stored would have the
        caller believe an image is indexed when it is not.
        """
        with pytest.raises(RuntimeError, match="not available"):
            await index.insert(np.ones(4))


class TestInterchangeableWithTheInMemoryIndex:
    """The router holds one or the other and never checks which."""

    def test_both_expose_the_same_async_surface(self) -> None:
        from api.services.similarity_index import SimilarityIndex

        required = {"insert", "query", "delete", "reset", "count", "snapshot"}
        memory = {m for m in required if callable(getattr(SimilarityIndex, m, None))}
        pg = {m for m in required if callable(getattr(PgVectorSimilarityIndex, m, None))}
        assert memory == required, f"SimilarityIndex is missing {required - memory}"
        assert pg == required, f"PgVectorSimilarityIndex is missing {required - pg}"

    async def test_snapshot_says_which_backend_is_live(self) -> None:
        """An operator reading /similarity/stats needs to know whether the
        index is shared. The two backends answer very differently under
        scaling, and nothing else in the response reveals which is in use."""
        from api.services.similarity_index import SimilarityIndex

        memory = await SimilarityIndex(dimension=4).snapshot()
        assert memory["backend"] == "memory"
        assert memory["shared"] is False
