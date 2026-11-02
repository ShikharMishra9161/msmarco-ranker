# Results

Generated from `reports/results.csv`.

| experiment                       |   ndcg_at_10 |   mrr_at_10 |   recall_at_100 |   latency_p50_ms |   latency_p99_ms |
|:---------------------------------|-------------:|------------:|----------------:|-----------------:|-----------------:|
| bm25                             |       0.5113 |      0.416  |          0.9263 |             2.3  |             5.15 |
| dense_flat                       |       0.6258 |      0.5153 |          1      |             9.2  |            15.25 |
| hybrid_bm25_dense_flat           |       0.6079 |      0.5133 |          1      |             9.2  |            15.25 |
| dense_ivf                        |       0.627  |      0.5167 |          1      |             8.99 |            11.41 |
| hybrid_bm25_dense_ivf            |       0.6117 |      0.5151 |          1      |             8.99 |            11.41 |
| rerank_no_finetuning_depth_20    |       0.7291 |      0.6568 |          1      |           355.41 |           546.81 |
| rerank_no_finetuning_depth_50    |       0.737  |      0.6608 |          1      |           802.93 |          1026.77 |
| rerank_random_negatives_depth_20 |       0.729  |      0.6568 |          1      |           369.9  |           514.55 |
| rerank_random_negatives_depth_50 |       0.7369 |      0.6608 |          1      |           910.98 |          1418.63 |
| rerank_hard_negatives_depth_20   |       0.7277 |      0.655  |          1      |           519.85 |           698.61 |
| rerank_hard_negatives_depth_50   |       0.7357 |      0.6591 |          1      |          1192.67 |          1519.45 |
