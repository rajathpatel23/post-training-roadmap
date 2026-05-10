"""
Prepare SFT training data for Project 1 — NER → fixed JSON schema (Path A).

Loads CoNLL-2003 (English) via the Hub parquet revision (works with recent
`datasets` where script-based loading is disabled).

Output:
  data/processed/sft_train.jsonl  — {"prompt", "response"}
  data/eval/sft_eval.jsonl        — {"prompt", "ground_truth"} (for exact-match eval)

Train/eval split: configurable fraction (default 80/20), fixed seed, disjoint indices.
Eval rows are sorted by original sentence order for a stable held-out file.

Run:
    python scripts/prepare_sft_data.py --config configs/sft/qwen05b_structured.yaml
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

from datasets import load_dataset

from src.common.config import ExperimentConfig
from src.common.io import write_jsonl
from src.common.prompts import (
    conll_tokens_and_tags_to_entities,
    entities_to_json_response,
    format_sft_ner_prompt,
)

# Parquet revision avoids deprecated Python loading scripts in datasets>=3.
_CONLL_REVISION = "refs/convert/parquet"


def _build_rows_from_conll(split_name: str = "train") -> list[dict[str, Any]]:
    ds = load_dataset("conll2003", revision=_CONLL_REVISION)
    hf_split = ds[split_name]
    names = hf_split.features["ner_tags"].feature.names

    rows: list[dict[str, Any]] = []
    for item in hf_split:
        tokens = item["tokens"]
        ner_strings = [names[int(i)] for i in item["ner_tags"]]
        sentence = " ".join(tokens)
        entities = conll_tokens_and_tags_to_entities(tokens, ner_strings)
        response = entities_to_json_response(entities)
        rows.append(
            {
                "prompt": format_sft_ner_prompt(sentence),
                "response": response,
            }
        )
    return rows


def main(config_path: str) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    extra = cfg.extra.get("sft_data") or {}
    seed = int(extra.get("seed", cfg.training.seed))
    train_fraction = float(extra.get("train_fraction", 0.8))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("sft_data.train_fraction must be in (0, 1)")
    max_examples = extra.get("max_examples")
    hf_split = str(extra.get("hf_split", "train"))

    records = _build_rows_from_conll(split_name=hf_split)
    if max_examples is not None:
        cap = int(max_examples)
        if cap < 2:
            raise ValueError("sft_data.max_examples must be at least 2 for train/eval split")
        rng = random.Random(seed)
        pick = sorted(rng.sample(range(len(records)), min(cap, len(records))))
        records = [records[i] for i in pick]

    n = len(records)
    if n < 2:
        raise ValueError("Need at least 2 examples after filtering")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(train_fraction * n)
    if n_train < 1 or n_train >= n:
        raise ValueError(
            f"train/eval split degenerate: n={n}, train_fraction={train_fraction} → n_train={n_train}"
        )

    train_ix = sorted(indices[:n_train])
    eval_ix = sorted(indices[n_train:])

    train_out = [{"prompt": records[i]["prompt"], "response": records[i]["response"]} for i in train_ix]
    eval_out = [
        {"prompt": records[i]["prompt"], "ground_truth": records[i]["response"]} for i in eval_ix
    ]

    train_path = os.path.abspath(cfg.data.train_path)
    eval_path = os.path.abspath(cfg.data.eval_path)
    write_jsonl(train_out, train_path)
    write_jsonl(eval_out, eval_path)

    print(
        f"Wrote {len(train_out)} train / {len(eval_out)} eval rows "
        f"({100 * len(train_out) / n:.1f}% / {100 * len(eval_out) / n:.1f}%) "
        f"from CoNLL-2003 split={hf_split!r} (seed={seed}).\n"
        f"  train: {train_path}\n"
        f"  eval:  {eval_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
