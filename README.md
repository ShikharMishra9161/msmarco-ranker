# msmarco-ranker

A reproducible, multi-stage MS MARCO passage-ranking portfolio project designed to run on a free Google Colab T4 GPU. It moves from BM25 to dense retrieval, reciprocal-rank fusion, and cross-encoder reranking.

## Pipeline

```mermaid
flowchart LR
  A[MS MARCO v1.1] --> B[Sample & clean]
  B --> C[Parquet corpus, queries, qrels]
  C --> D[BM25 top-100]
  C --> E[MiniLM + FAISS]
  D --> F[RRF hybrid top-100]
  E --> F
  F --> G[Cross-encoder rerank]
  G --> H[NDCG@10, MRR@10, Recall@100, latency]
```

## Dataset and cleaning

The data script streams Hugging Face `microsoft/ms_marco` (`v1.1`, the current namespaced form of `ms_marco`), reservoir-samples configurable train and validation subsets, and saves parquet files. It removes empty or very short passages, non-English passages, exact normalized duplicates, MinHash near duplicates, and dev queries that no longer have a relevant retained passage. The precise filter counts are generated—not assumed—in [reports/data_quality.md](reports/data_quality.md) after preparation.

## Run on Colab

```bash
git clone <your-repository-url> && cd msmarco-ranker
pip install -r requirements.txt
python scripts/prepare_data.py --config configs/default.yaml
pytest -q
python scripts/run_bm25.py --config configs/default.yaml
python scripts/run_dense.py --config configs/default.yaml
python scripts/run_reranker.py --config configs/default.yaml
python scripts/generate_report.py
```

Set Colab's runtime accelerator to T4 GPU before dense encoding and reranking. All sizes, paths, models, seeds, retrieval settings, and training settings live in `configs/default.yaml`. The default 500K training sample is intentionally sizeable; reduce `data.train_pairs` while debugging.

## Results and ablations

Every completed run appends its full YAML config plus NDCG@10, MRR@10, Recall@100, and p50/p99 per-query latency to `reports/results.csv`. Run `scripts/generate_report.py` to render the current table in [reports/results.md](reports/results.md). Until experiments execute, metrics are intentionally absent.

Reranker ablations cover no fine-tuning, random negatives, BM25 hard negatives, and rerank depths 20/50/100. Use `scripts/worst_queries.py --rankings <file.csv>` after exporting rankings to print the 20 lowest-NDCG queries for error analysis.
