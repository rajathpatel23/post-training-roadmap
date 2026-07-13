"""
DPO data loading: tokenize (prompt, chosen, rejected) triples, reusing
sft_bpe_data.py's build_example() for each of chosen/rejected — same
prompt-masking convention (loss/log-prob only over response tokens, prompt
conditions but isn't scored) since DPO needs exactly the same
"log-prob of the response given the prompt" quantity SFT's loss already
isolates, just for two candidate responses instead of one ground truth.
"""

import json
import random

from sft_bpe_data import build_example, collate


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize_examples(tokenizer, raw_examples: list[dict], block_size: int) -> list[dict]:
    """Tokenize once, reused across every epoch/eval call — same reasoning
    as sft_bpe_data.py's tokenize_examples."""
    return [
        {
            "chosen": build_example(tokenizer, e["prompt"], e["chosen"], block_size),
            "rejected": build_example(tokenizer, e["prompt"], e["rejected"], block_size),
        }
        for e in raw_examples
    ]


def iterate_epoch(examples: list[dict], tokenizer, block_size: int, batch_size: int, device: str, shuffle: bool = True):
    """Yields (chosen_batch, rejected_batch) pairs, each a (input_ids,
    labels, attention_mask) triple from sft_bpe_data.py's collate() —
    chosen and rejected are batched/padded independently since they're
    different lengths, not concatenated into one tensor."""
    if shuffle:
        examples = examples.copy()
        random.shuffle(examples)

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        if not batch:
            continue
        chosen_batch = collate(tokenizer, [e["chosen"] for e in batch], block_size, device)
        rejected_batch = collate(tokenizer, [e["rejected"] for e in batch], block_size, device)
        yield chosen_batch, rejected_batch
