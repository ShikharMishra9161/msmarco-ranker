"""Print the 20 lowest-NDCG queries for qualitative error analysis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.evaluate import qrels_by_query
from src.eval.metrics import ndcg_at_k


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--rankings", required=True, help="CSV with query_id and ranked_passage_ids JSON column")
    parser.add_argument("--queries", default="data/processed/dev_queries.parquet"); parser.add_argument("--qrels", default="data/processed/dev_qrels.parquet")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    ranks = pd.read_csv(root / args.rankings); queries = pd.read_parquet(root / args.queries); rel = qrels_by_query(pd.read_parquet(root / args.qrels))
    import json
    ranks["ndcg_at_10"] = ranks.apply(lambda row: ndcg_at_k(json.loads(row.ranked_passage_ids), rel.get(row.query_id, {})), axis=1)
    print(ranks.merge(queries, on="query_id").nsmallest(20, "ndcg_at_10")[["query_id", "query", "ndcg_at_10"]].to_string(index=False))


if __name__ == "__main__": main()
