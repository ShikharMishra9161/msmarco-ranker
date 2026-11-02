"""Shared reproducibility, configuration, and reporting helpers."""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    """Create and return project-relative configured directories."""
    paths = {name: root / value for name, value in config["paths"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def append_result(report_dir: Path, experiment: str, config: dict[str, Any], metrics: dict[str, float]) -> None:
    """Append a measured experiment and its complete configuration to CSV."""
    target = report_dir / "results.csv"
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "experiment": experiment,
           "config_json": json.dumps(config, sort_keys=True), **metrics}
    write_header = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_rankings(path: Path | str, rankings: dict[str, list[str]]) -> None:
    """Save query rankings to CSV with query_id and ranked_passage_ids JSON column."""
    import pandas as pd
    rows = [{"query_id": qid, "ranked_passage_ids": json.dumps(list(pids))} for qid, pids in rankings.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def load_rankings(path: Path | str) -> dict[str, list[str]]:
    """Load query rankings from CSV saved with save_rankings."""
    import pandas as pd
    df = pd.read_csv(path)
    return {str(row.query_id): json.loads(row.ranked_passage_ids) for row in df.itertuples(index=False)}

