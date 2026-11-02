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

You can run the entire pipeline directly in Google Colab using `notebooks/colab_run.ipynb`:
1. Open Google Colab and upload `notebooks/colab_run.ipynb`.
2. Change the runtime accelerator to **T4 GPU** (**Runtime → Change runtime type → T4 GPU**).
3. Upload `msmarco-ranker-colab.zip` when prompted in Step 2.
4. Run all cells sequentially.

Alternatively, execute via CLI:

```bash
git clone https://github.com/ShikharMishra9161/msmarco-ranker && cd msmarco-ranker
pip install -r requirements.txt
pytest -v
python scripts/prepare_data.py --config configs/default.yaml
python scripts/run_bm25.py --config configs/default.yaml
python scripts/run_dense.py --config configs/default.yaml
python scripts/run_reranker.py --config configs/default.yaml
python scripts/generate_report.py
```

Set Colab's runtime accelerator to T4 GPU before dense encoding and reranking. All sizes, paths, models, seeds, retrieval settings, and training settings live in `configs/default.yaml`. For a fast 2-minute smoke test, use `configs/debug.yaml`.


## Results and ablations

Every completed run appends its full YAML config plus NDCG@10, MRR@10, Recall@100, and p50/p99 per-query latency to `reports/results.csv`. Run `scripts/generate_report.py` to render the current table in [reports/results.md](reports/results.md).

Current results (Colab T4; 95 dev queries over a cleaned corpus of 2,459 passages, ~200 training queries, retrieval depth 50):

| Stage | NDCG@10 | MRR@10 | p50 latency (ms) | p99 latency (ms) |
|:--|--:|--:|--:|--:|
| BM25 | 0.5113 | 0.4160 | 2.3 | 5.2 |
| Dense (MiniLM, FAISS FlatIP) | 0.6258 | 0.5153 | 9.2 | 15.3 |
| Dense (MiniLM, FAISS IVF-Flat) | 0.6270 | 0.5167 | 9.0 | 11.4 |
| Hybrid BM25 + dense IVF (RRF) | 0.6117 | 0.5151 | 9.0 | 11.4 |
| Cross-encoder rerank, depth 20 | 0.7291 | 0.6568 | 355 | 547 |
| **Cross-encoder rerank, depth 50** | **0.7370** | **0.6608** | 803 | 1,027 |
| + fine-tuned, random negatives (depth 50) | 0.7369 | 0.6608 | 911 | 1,419 |
| + fine-tuned, BM25 hard negatives (depth 50) | 0.7357 | 0.6591 | 1,193 | 1,519 |

Findings:

- Dense retrieval beats BM25 by 22.6% NDCG@10. IVF-Flat matches exact search quality with 25% lower p99 latency.
- Cross-encoder reranking gives the largest gain: +44.1% NDCG@10 over BM25. Going from depth 20 to 50 adds only +0.008 NDCG at 2.3x the p50 latency.
- RRF fusion scores below dense-only retrieval here; BM25 adds more noise than complementary signal on this subset.
- Fine-tuning the already MS MARCO-trained cross-encoder on ~200 queries does not help, with either random or hard negatives. A larger training set is the next experiment.
- Recall@100 is 1.0 for dense and reranked runs only because the evaluation corpus is small, so it is not a meaningful metric at this scale.

Data cleaning on this run removed 5 exact duplicates, 21 MinHash near duplicates and 5 dev queries left without a relevant passage; see [reports/data_quality.md](reports/data_quality.md).

Reranker ablations cover no fine-tuning, random negatives, BM25 hard negatives, and rerank depths 20/50 (100 is configurable). Use `scripts/worst_queries.py --rankings <file.csv>` after exporting rankings to print the 20 lowest-NDCG queries for error analysis.
