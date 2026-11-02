"""Evaluation and latency aggregation for ranked retrieval runs."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


def qrels_by_query(qrels: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Convert qrel dataframe to query-keyed relevance mappings."""
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for row in qrels.itertuples(index=False):
        output[str(row.query_id)][str(row.passage_id)] = float(row.relevance)
    return dict(output)


def evaluate_rankings(rankings: Mapping[str, Sequence[str]], qrels: pd.DataFrame, latencies_ms: Sequence[float]) -> dict[str, float]:
    """Calculate mean ranking metrics and p50/p99 query latency."""
    grouped = qrels_by_query(qrels)
    return {
        "ndcg_at_10": float(np.mean([ndcg_at_k(rankings.get(qid, []), rel, 10) for qid, rel in grouped.items()])),
        "mrr_at_10": float(np.mean([mrr_at_k(rankings.get(qid, []), rel, 10) for qid, rel in grouped.items()])),
        "recall_at_100": float(np.mean([recall_at_k(rankings.get(qid, []), rel, 100) for qid, rel in grouped.items()])),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p99_ms": float(np.percentile(latencies_ms, 99)),
    }
