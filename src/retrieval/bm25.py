"""BM25 lexical retrieval over a cleaned passage corpus."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Simple in-memory BM25 retriever with stable output identifiers."""

    def __init__(self, corpus: pd.DataFrame, k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = corpus["passage_id"].astype(str).tolist()
        self._bm25 = BM25Okapi([text.lower().split() for text in corpus["passage"].astype(str)], k1=k1, b=b)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return top documents and BM25 scores for one query."""
        scores = self._bm25.get_scores(query.lower().split())
        indexes = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.ids[int(index)], float(scores[index])) for index in indexes]

    def batch_search(self, queries: Sequence[str], top_k: int = 100) -> list[list[tuple[str, float]]]:
        """Retrieve independently for a batch of queries."""
        return [self.search(query, top_k) for query in queries]
