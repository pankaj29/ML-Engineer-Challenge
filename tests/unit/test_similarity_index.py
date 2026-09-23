"""Unit tests for the similarity vector index."""

from __future__ import annotations

import numpy as np
import pytest

from api.services.similarity_index import SimilarityIndex


@pytest.fixture
def index() -> SimilarityIndex:
    return SimilarityIndex(dimension=8)


def unit(*values: float) -> np.ndarray:
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class TestAdd:
    def test_add_returns_id_and_grows(self, index: SimilarityIndex) -> None:
        item_id = index.add(np.ones(8))
        assert isinstance(item_id, str)
        assert index.size == 1

    def test_uses_supplied_id(self, index: SimilarityIndex) -> None:
        assert index.add(np.ones(8), item_id="my-id") == "my-id"

    def test_stores_label_and_metadata(self, index: SimilarityIndex) -> None:
        index.add(np.ones(8), label="a cat", metadata={"source": "upload"})
        item, _ = index.search(np.ones(8), top_k=1)[0]
        assert item.label == "a cat"
        assert item.metadata == {"source": "upload"}

    def test_normalises_on_insert(self, index: SimilarityIndex) -> None:
        """An un-normalised vector would corrupt every score computed against it."""
        index.add(np.array([100.0, 0, 0, 0, 0, 0, 0, 0]))
        assert np.linalg.norm(index._vectors[0]) == pytest.approx(1.0)

    def test_rejects_wrong_dimension(self, index: SimilarityIndex) -> None:
        with pytest.raises(ValueError, match="8-dimensional"):
            index.add(np.ones(16))

    def test_add_many(self, index: SimilarityIndex) -> None:
        ids = index.add_many(np.random.randn(5, 8))
        assert len(ids) == 5
        assert index.size == 5

    def test_add_many_rejects_wrong_shape(self, index: SimilarityIndex) -> None:
        with pytest.raises(ValueError):
            index.add_many(np.random.randn(5, 16))

    def test_add_many_normalises(self, index: SimilarityIndex) -> None:
        index.add_many(np.random.randn(5, 8) * 100)
        np.testing.assert_allclose(np.linalg.norm(index._vectors, axis=1), 1.0, atol=1e-5)


class TestSearch:
    def test_finds_exact_match_at_score_one(self, index: SimilarityIndex) -> None:
        query = unit(1, 0, 0, 0, 0, 0, 0, 0)
        index.add(query, label="target")
        results = index.search(query, top_k=1)
        assert results[0][0].label == "target"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_orders_by_similarity(self, index: SimilarityIndex) -> None:
        index.add(unit(1, 0, 0, 0, 0, 0, 0, 0), label="identical")
        index.add(unit(1, 1, 0, 0, 0, 0, 0, 0), label="close")
        index.add(unit(0, 0, 0, 0, 0, 0, 0, 1), label="orthogonal")

        results = index.search(unit(1, 0, 0, 0, 0, 0, 0, 0), top_k=3)
        assert [item.label for item, _ in results] == ["identical", "close", "orthogonal"]
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_opposite_vector_scores_minus_one(self, index: SimilarityIndex) -> None:
        index.add(unit(1, 0, 0, 0, 0, 0, 0, 0))
        _, score = index.search(unit(-1, 0, 0, 0, 0, 0, 0, 0), top_k=1)[0]
        assert score == pytest.approx(-1.0, abs=1e-5)

    def test_min_similarity_filters(self, index: SimilarityIndex) -> None:
        index.add(unit(1, 0, 0, 0, 0, 0, 0, 0))
        index.add(unit(0, 0, 0, 0, 0, 0, 0, 1))
        results = index.search(unit(1, 0, 0, 0, 0, 0, 0, 0), top_k=10, min_similarity=0.5)
        assert len(results) == 1

    def test_top_k_caps_results(self, index: SimilarityIndex) -> None:
        index.add_many(np.random.randn(20, 8))
        assert len(index.search(np.random.randn(8), top_k=3)) == 3

    def test_top_k_larger_than_index(self, index: SimilarityIndex) -> None:
        index.add_many(np.random.randn(2, 8))
        assert len(index.search(np.random.randn(8), top_k=100)) == 2

    def test_empty_index_returns_nothing(self, index: SimilarityIndex) -> None:
        assert index.search(np.random.randn(8)) == []

    def test_rejects_wrong_query_dimension(self, index: SimilarityIndex) -> None:
        index.add(np.ones(8))
        with pytest.raises(ValueError, match="dimensions"):
            index.search(np.ones(16))

    def test_normalises_unnormalised_query(self, index: SimilarityIndex) -> None:
        index.add(unit(1, 0, 0, 0, 0, 0, 0, 0))
        _, score = index.search(np.array([50.0, 0, 0, 0, 0, 0, 0, 0]), top_k=1)[0]
        assert score == pytest.approx(1.0, abs=1e-5)


class TestRemoveAndClear:
    def test_remove_existing(self, index: SimilarityIndex) -> None:
        item_id = index.add(np.ones(8))
        assert index.remove(item_id) is True
        assert index.size == 0

    def test_remove_missing_returns_false(self, index: SimilarityIndex) -> None:
        assert index.remove("not-there") is False

    def test_remove_keeps_vectors_and_items_aligned(self, index: SimilarityIndex) -> None:
        """A misaligned remove would attribute scores to the wrong image."""
        index.add(unit(1, 0, 0, 0, 0, 0, 0, 0), item_id="a", label="A")
        index.add(unit(0, 1, 0, 0, 0, 0, 0, 0), item_id="b", label="B")
        index.add(unit(0, 0, 1, 0, 0, 0, 0, 0), item_id="c", label="C")

        index.remove("b")

        assert index.size == 2
        assert index._vectors.shape[0] == 2
        item, score = index.search(unit(0, 0, 1, 0, 0, 0, 0, 0), top_k=1)[0]
        assert item.label == "C"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_clear(self, index: SimilarityIndex) -> None:
        index.add_many(np.random.randn(5, 8))
        index.clear()
        assert index.size == 0
        assert index.search(np.random.randn(8)) == []


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path) -> None:
        source = SimilarityIndex(dimension=8)
        source.add(unit(1, 0, 0, 0, 0, 0, 0, 0), item_id="x", label="labelled", metadata={"k": 1})
        source.add(unit(0, 1, 0, 0, 0, 0, 0, 0), item_id="y")

        path = source.save(tmp_path / "index.npz")

        restored = SimilarityIndex(dimension=8)
        assert restored.load(path) is True
        assert restored.size == 2

        item, score = restored.search(unit(1, 0, 0, 0, 0, 0, 0, 0), top_k=1)[0]
        assert item.id == "x"
        assert item.label == "labelled"
        assert item.metadata == {"k": 1}
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_load_missing_file_returns_false(self, tmp_path) -> None:
        assert SimilarityIndex(dimension=8).load(tmp_path / "nope.npz") is False

    def test_inconsistent_files_are_rejected(self, tmp_path) -> None:
        """Mismatched vector and metadata counts must not serve wrong results."""
        import json

        source = SimilarityIndex(dimension=8)
        source.add_many(np.random.randn(3, 8))
        path = source.save(tmp_path / "index.npz")

        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta["items"] = meta["items"][:1]  # corrupt: 3 vectors, 1 item
        meta_path.write_text(json.dumps(meta))

        restored = SimilarityIndex(dimension=8)
        assert restored.load(path) is False
        assert restored.size == 0


class TestStats:
    def test_stats_report(self, index: SimilarityIndex) -> None:
        index.add_many(np.random.randn(10, 8))
        stats = index.stats()
        assert stats["size"] == 10
        assert stats["dimension"] == 8
        assert stats["memory_mb"] >= 0

    def test_len(self, index: SimilarityIndex) -> None:
        index.add_many(np.random.randn(4, 8))
        assert len(index) == 4
