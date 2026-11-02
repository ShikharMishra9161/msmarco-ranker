"""Cross-encoder training, candidate mining, and reranking."""
from __future__ import annotations

import random
from collections.abc import Sequence

import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader


def mine_training_examples(queries: pd.DataFrame, qrels: pd.DataFrame, corpus: pd.DataFrame,
                           bm25_results: dict[str, list[str]], negative_mode: str, negatives: int, seed: int) -> list[InputExample]:
    """Create positive pairs and random or BM25 hard-negative pairs."""
    rng = random.Random(seed)
    text = dict(zip(corpus.passage_id.astype(str), corpus.passage))
    all_ids = list(text)
    relevant = qrels.groupby("query_id")["passage_id"].apply(lambda ids: set(ids.astype(str))).to_dict()
    examples: list[InputExample] = []
    for query in queries.itertuples(index=False):
        positives = relevant.get(query.query_id, set())
        for passage_id in positives:
            if passage_id in text:
                examples.append(InputExample(texts=[query.query, text[passage_id]], label=1.0))
        candidates = [doc for doc in bm25_results.get(query.query_id, []) if doc not in positives]
        if negative_mode == "random":
            candidates = [doc for doc in all_ids if doc not in positives]
            rng.shuffle(candidates)
        for passage_id in candidates[:negatives]:
            examples.append(InputExample(texts=[query.query, text[passage_id]], label=0.0))
    return examples


def fine_tune(model_name: str, examples: Sequence[InputExample], batch_size: int, epochs: int, learning_rate: float, output_dir: str) -> CrossEncoder:
    """Fine-tune a cross-encoder on labelled query-passage pairs."""
    model = CrossEncoder(model_name, num_labels=1, max_length=256)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model.fit(train_dataloader=loader, epochs=epochs, optimizer_params={"lr": learning_rate}, output_path=output_dir,
              show_progress_bar=True)
    return CrossEncoder(output_dir, max_length=256)


def rerank(model: CrossEncoder, query: str, candidate_ids: Sequence[str], passage_text: dict[str, str], depth: int) -> list[str]:
    """Score top candidate depth and retain remaining candidates in prior order."""
    head = list(candidate_ids[:depth])
    scores = model.predict([(query, passage_text[item]) for item in head], show_progress_bar=False)
    return [item for item, _ in sorted(zip(head, scores), key=lambda pair: -float(pair[1]))] + list(candidate_ids[depth:])
