"""Hand-worked regression tests for ranking metrics."""
import pytest
import math

from src.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_perfect_ranking() -> None:
    qrels = {"a": 3, "b": 1}
    assert ndcg_at_k(["a", "b"], qrels, 10) == pytest.approx(1.0)
    assert mrr_at_k(["a", "b"], qrels, 10) == pytest.approx(1.0)
    assert recall_at_k(["a", "b"], qrels, 100) == pytest.approx(1.0)


def test_known_imperfect_ranking() -> None:
    qrels = {"a": 1, "b": 1}
    ranked = ["x", "a", "b"]
    assert mrr_at_k(ranked, qrels, 10) == pytest.approx(0.5)
    assert recall_at_k(ranked, qrels, 2) == pytest.approx(0.5)
    expected = (1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(expected)


def test_reciprocal_rank_fusion() -> None:
    from src.retrieval.dense import reciprocal_rank_fusion
    # List 1: docA (rank 1), docB (rank 2)
    # List 2: docB (rank 1), docA (rank 2)
    l1 = [("docA", 1.0), ("docB", 0.5)]
    l2 = [("docB", 1.0), ("docA", 0.5)]
    fused = reciprocal_rank_fusion([l1, l2], rrf_k=60, top_k=2)
    assert set(fused) == {"docA", "docB"}


def test_save_load_rankings(tmp_path) -> None:
    from src.utils import save_rankings, load_rankings
    test_file = tmp_path / "test_ranks.csv"
    rankings = {"q1": ["doc1", "doc2"], "q2": ["doc3"]}
    save_rankings(test_file, rankings)
    loaded = load_rankings(test_file)
    assert loaded == rankings

