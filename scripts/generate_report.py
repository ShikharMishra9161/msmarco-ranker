"""Render the current experiment CSV into a Markdown results report."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "reports" / "results.csv"
    target = root / "reports" / "results.md"
    if not source.exists():
        target.write_text("# Results\n\nNo experiments have been run yet; no metrics are reported.\n", encoding="utf-8")
        return
    frame = pd.read_csv(source)
    if "experiment" in frame:
        frame = frame.drop_duplicates(subset=["experiment"], keep="last")
    metric_cols = ["ndcg_at_10", "mrr_at_10", "recall_at_100"]
    latency_cols = ["latency_p50_ms", "latency_p99_ms"]
    for col in metric_cols:
        if col in frame:
            frame[col] = frame[col].round(4)
    for col in latency_cols:
        if col in frame:
            frame[col] = frame[col].round(2)
    columns = [column for column in ["experiment", *metric_cols, *latency_cols] if column in frame]
    table_md = frame[columns].to_markdown(index=False)
    content = f"# Results\n\nGenerated from `reports/results.csv`.\n\n{table_md}\n"
    target.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__": main()
