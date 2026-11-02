"""Download, clean, and persist the configured MS MARCO subset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cleaning import clean_passages, write_quality_report
from src.data.loading import flatten_examples, load_ms_marco_samples
from src.utils import load_config, resolve_paths, set_seed


def main() -> None:
    """Execute deterministic data preparation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    set_seed(config["seed"])
    paths = resolve_paths(config, root)
    train_raw, dev_raw = load_ms_marco_samples(config)
    train_q, train_p, train_qrels = flatten_examples(train_raw, "train", config["data"]["max_passages_per_query"])
    dev_q, dev_p, dev_qrels = flatten_examples(dev_raw, "dev", config["data"]["max_passages_per_query"])
    all_passages = pd.concat([train_p, dev_p], ignore_index=True)
    clean, mapping, stats = clean_passages(all_passages, config["data"]["min_passage_tokens"], config["data"]["language"], config["data"]["duplicate_jaccard_threshold"])
    dev_qrels = dev_qrels.merge(mapping, on="original_id", how="inner")
    dev_qrels = dev_qrels[["query_id", "passage_id", "relevance"]].drop_duplicates()
    valid_query_ids = set(dev_qrels.query_id)
    removed_queries = len(dev_q) - len(valid_query_ids)
    dev_q = dev_q[dev_q.query_id.isin(valid_query_ids)]
    train_qrels = train_qrels.merge(mapping, on="original_id", how="inner")
    train_qrels = train_qrels[["query_id", "passage_id", "relevance"]].drop_duplicates()
    train_q = train_q[train_q.query_id.isin(set(train_qrels.query_id))]
    output = paths["processed_dir"]
    clean.to_parquet(output / "corpus.parquet", index=False)
    train_q.to_parquet(output / "train_queries.parquet", index=False)
    train_qrels.to_parquet(output / "train_qrels.parquet", index=False)
    dev_q.to_parquet(output / "dev_queries.parquet", index=False)
    dev_qrels.to_parquet(output / "dev_qrels.parquet", index=False)
    write_quality_report(str(paths["reports_dir"] / "data_quality.md"), stats, len(all_passages), removed_queries, len(clean))
    print(f"Saved {len(clean):,} passages, {len(train_q):,} training queries, {len(dev_q):,} dev queries.")


if __name__ == "__main__":
    main()
