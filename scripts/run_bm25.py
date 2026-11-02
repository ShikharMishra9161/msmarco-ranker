"""Evaluate BM25 retrieval and log measured metrics."""
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
    print(f"Building BM25 index over {len(corpus):,} passages...")
    retriever = BM25Retriever(corpus, config["bm25"]["k1"], config["bm25"]["b"])
    rankings, times = {}, []
    for row in tqdm(queries.itertuples(index=False), total=len(queries), desc="Evaluating BM25"):
        start = time.perf_counter()
        result = retriever.search(row.query, config["bm25"]["top_k"])
        times.append((time.perf_counter() - start) * 1000)
        rankings[row.query_id] = [doc_id for doc_id, _ in result]
    metrics = evaluate_rankings(rankings, qrels, times)
    save_rankings(paths["artifacts_dir"] / "rankings_bm25.csv", rankings)
    append_result(paths["reports_dir"], "bm25", config, metrics)
    print("BM25 results:", metrics)


if __name__ == "__main__":
    main()
