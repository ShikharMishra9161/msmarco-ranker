"""Compare FlatIP, IVF-Flat, and RRF hybrid retrieval."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.evaluate import evaluate_rankings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, reciprocal_rank_fusion
from src.utils import append_result, load_config, resolve_paths, save_rankings, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    set_seed(config["seed"])
    paths = resolve_paths(config, root)
    corpus = pd.read_parquet(paths["processed_dir"] / "corpus.parquet")
    queries = pd.read_parquet(paths["processed_dir"] / "dev_queries.parquet")
    qrels = pd.read_parquet(paths["processed_dir"] / "dev_qrels.parquet")

    print(f"Building BM25 cache for hybrid fusion over {len(queries):,} dev queries...")
    bm25 = BM25Retriever(corpus, config["bm25"]["k1"], config["bm25"]["b"])
    bm25_searches = {row.query_id: bm25.search(row.query, 100) for row in queries.itertuples(index=False)}

    cached_vectors = None
    for kind in ("flat", "ivf"):
        print(f"\nSetting up Dense Index ({kind.upper()})...")
        dense = DenseRetriever(corpus, config["dense"]["model_name"], config["dense"]["batch_size"],
                               kind, config["dense"]["ivf_nlist"], config["dense"]["ivf_nprobe"],
                               vectors=cached_vectors)
        cached_vectors = dense.vectors
        dense.save(str(paths["artifacts_dir"] / f"faiss_{kind}.index"))

        ranks, hybrid, times = {}, {}, []
        for row in tqdm(queries.itertuples(index=False), total=len(queries), desc=f"Dense {kind} retrieval"):
            start = time.perf_counter()
            dense_result = dense.search(row.query, config["dense"]["top_k"])
            times.append((time.perf_counter() - start) * 1000)
            ranks[row.query_id] = [doc for doc, _ in dense_result]
            hybrid[row.query_id] = reciprocal_rank_fusion([bm25_searches[row.query_id], dense_result], config["hybrid"]["rrf_k"], 100)

        dense_metrics = evaluate_rankings(ranks, qrels, times)
        hybrid_metrics = evaluate_rankings(hybrid, qrels, times)
        save_rankings(paths["artifacts_dir"] / f"rankings_dense_{kind}.csv", ranks)
        save_rankings(paths["artifacts_dir"] / f"rankings_hybrid_{kind}.csv", hybrid)
        append_result(paths["reports_dir"], f"dense_{kind}", config, dense_metrics)
        append_result(paths["reports_dir"], f"hybrid_bm25_dense_{kind}", config, hybrid_metrics)
        print(f"dense_{kind}:", dense_metrics)
        print(f"hybrid_bm25_dense_{kind}:", hybrid_metrics)


if __name__ == "__main__": main()
