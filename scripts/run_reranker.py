"""Run requested cross-encoder training and reranking ablations."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.evaluate import evaluate_rankings
from src.rerank.cross_encoder import fine_tune, mine_training_examples, rerank
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, reciprocal_rank_fusion
from src.utils import append_result, load_config, load_rankings, resolve_paths, save_rankings, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    set_seed(config["seed"])
    paths = resolve_paths(config, root)
    corpus = pd.read_parquet(paths["processed_dir"] / "corpus.parquet")
    train_q = pd.read_parquet(paths["processed_dir"] / "train_queries.parquet")
    train_rel = pd.read_parquet(paths["processed_dir"] / "train_qrels.parquet")
    dev_q = pd.read_parquet(paths["processed_dir"] / "dev_queries.parquet")
    dev_rel = pd.read_parquet(paths["processed_dir"] / "dev_qrels.parquet")

    # Load precomputed hybrid rankings from run_dense.py if present, avoiding re-encoding
    hybrid_cache_file = paths["artifacts_dir"] / "rankings_hybrid_flat.csv"
    if hybrid_cache_file.exists():
        print(f"Loading cached hybrid candidate pool from {hybrid_cache_file}...")
        hybrid = load_rankings(hybrid_cache_file)
    else:
        print("Computing BM25 & Dense retrieval for hybrid candidate pool...")
        bm25 = BM25Retriever(corpus, config["bm25"]["k1"], config["bm25"]["b"])
        flat_index_path = paths["artifacts_dir"] / "faiss_flat.index"
        if flat_index_path.exists():
            dense = DenseRetriever.from_index(str(flat_index_path), config["dense"]["model_name"], corpus.passage_id.astype(str).tolist())
        else:
            dense = DenseRetriever(corpus, config["dense"]["model_name"], config["dense"]["batch_size"])
        hybrid = {
            row.query_id: reciprocal_rank_fusion([bm25.search(row.query, 100), dense.search(row.query, 100)], config["hybrid"]["rrf_k"], 100)
            for row in tqdm(dev_q.itertuples(index=False), total=len(dev_q), desc="Hybrid retrieval")
        }

    text = dict(zip(corpus.passage_id.astype(str), corpus.passage))
    variants = [("no_finetuning", None), ("random_negatives", "random"), ("hard_negatives", "hard")]
    train_bm25 = None

    for label, mode in variants:
        target_files = [paths["artifacts_dir"] / f"rankings_{label}_depth_{d}.csv" for d in config["reranker"]["rerank_depths"]]
        if all(f.exists() for f in target_files):
            print(f"\n=== Variant {label} already evaluated, skipping... ===")
            continue

        print(f"\n=== Running Reranker Variant: {label} (mode={mode}) ===")
        if mode is None:
            model = CrossEncoder(config["reranker"]["model_name"], max_length=config["reranker"]["max_length"])
        else:
            if mode == "hard" and train_bm25 is None:
                print("Mining BM25 candidates for training queries...")
                bm25_miner = BM25Retriever(corpus, config["bm25"]["k1"], config["bm25"]["b"])
                train_bm25 = {
                    row.query_id: [item[0] for item in bm25_miner.search(row.query, 100)]
                    for row in tqdm(train_q.itertuples(index=False), total=len(train_q), desc="Train BM25 mining")
                }
            examples = mine_training_examples(train_q, train_rel, corpus, train_bm25 or {}, mode, config["reranker"]["hard_negative_count"], config["seed"])
            print(f"Fine-tuning {config['reranker']['model_name']} on {len(examples):,} mined pairs...")
            model = fine_tune(config["reranker"]["model_name"], examples, config["reranker"]["batch_size"], config["reranker"]["epochs"], config["reranker"]["learning_rate"], str(paths["artifacts_dir"] / f"reranker_{label}"))

        for depth in config["reranker"]["rerank_depths"]:
            print(f"Reranking top-{depth} candidates across {len(dev_q):,} queries...")
            rankings, timings = {}, []
            for row in tqdm(dev_q.itertuples(index=False), total=len(dev_q), desc=f"Rerank {label} depth {depth}"):
                start = time.perf_counter()
                rankings[row.query_id] = rerank(model, row.query, hybrid[row.query_id], text, depth)
                timings.append((time.perf_counter() - start) * 1000)

            metrics = evaluate_rankings(rankings, dev_rel, timings)
            save_rankings(paths["artifacts_dir"] / f"rankings_{label}_depth_{depth}.csv", rankings)
            append_result(paths["reports_dir"], f"rerank_{label}_depth_{depth}", config, metrics)
            print(f"rerank_{label}_depth_{depth}:", metrics)


if __name__ == "__main__": main()
