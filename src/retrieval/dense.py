"""SentenceTransformer + FAISS dense retrieval and reciprocal-rank fusion."""
from __future__ import annotations

from collections.abc import Sequence

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """FAISS inner-product index with FlatIP and IVF-Flat choices."""

    def __init__(self, corpus: pd.DataFrame, model_name: str, batch_size: int, index_type: str = "flat", nlist: int = 256, nprobe: int = 16, vectors: np.ndarray | None = None) -> None:
        self.ids = corpus.passage_id.astype(str).tolist()
        self.model = SentenceTransformer(model_name)
        if vectors is None:
            vectors = self.model.encode(corpus.passage.tolist(), batch_size=batch_size, normalize_embeddings=True,
                                        show_progress_bar=True, convert_to_numpy=True).astype("float32")
        self.vectors = vectors
        dim = vectors.shape[1]
        if index_type == "ivf":
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, min(nlist, len(vectors)), faiss.METRIC_INNER_PRODUCT)
            self.index.train(vectors)
            self.index.nprobe = min(nprobe, self.index.nlist)
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    @classmethod
    def from_index(cls, index_path: str, model_name: str, ids: Sequence[str]) -> "DenseRetriever":
        """Load an existing FAISS index without re-encoding the corpus."""
        instance = cls.__new__(cls)
        instance.ids = list(ids)
        instance.model = SentenceTransformer(model_name)
        instance.index = faiss.read_index(index_path)
        instance.vectors = None
        return instance

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Encode and retrieve one query."""
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, positions = self.index.search(vector, top_k)
        return [(self.ids[int(position)], float(score)) for position, score in zip(positions[0], scores[0]) if position >= 0]

    def save(self, path: str) -> None:
        """Persist the FAISS index for repeatable inference deployments."""
        faiss.write_index(self.index, path)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[tuple[str, float]]], rrf_k: int = 60, top_k: int = 100) -> list[str]:
    """Fuse ranked lists using reciprocal rank fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
