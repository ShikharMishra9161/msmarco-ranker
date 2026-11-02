"""Standard ranking metrics implemented without benchmark-specific dependencies."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int | float], k: int = 10) -> float:
    """Compute graded NDCG at k for one query."""
    if not qrels:
        return 0.0
    dcg = sum((2.0 ** float(qrels.get(doc_id, 0)) - 1.0) / math.log2(rank + 2)
              for rank, doc_id in enumerate(ranked_ids[:k]))
    ideal = sorted((float(value) for value in qrels.values()), reverse=True)[:k]
    idcg = sum((2.0 ** value - 1.0) / math.log2(rank + 2) for rank, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int | float], k: int = 10) -> float:
    """Return reciprocal rank of the first relevant document at k."""
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, int | float], k: int = 100) -> float:
    """Return fraction of relevant documents recovered at k."""
    relevant = {doc_id for doc_id, value in qrels.items() if value > 0}
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant) if relevant else 0.0
