"""
SFT data loading: tokenize (prompt, response) pairs from sft_data.py's JSONL
output with the BPE tokenizer, build the loss mask (prompt tokens excluded,
response tokens + EOS included), pad for batching.

Unlike pretraining's infinite random-window sampling (data.py/bpe_data.py's
get_batch), SFT trains over a small, fixed dataset for a few epochs — so
this iterates the actual example list, not a random 1D token stream.
"""

import json
import random

import torch

IGNORE_INDEX = -100


def load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_example(tokenizer, prompt: str, response: str, block_size: int) -> dict:
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + [tokenizer.eos_id]

    full_ids = (prompt_ids + response_ids)[:block_size]
    n_prompt = min(len(prompt_ids), len(full_ids))

    # Standard causal-LM shift: the model's output at position i predicts
    # full_ids[i+1], not full_ids[i] — this is what get_batch() in data.py
    # already does for pretraining (y = data[i+1:i+1+block_size]). Skipping
    # this shift trains the model to predict its own current input token,
    # a trivial/degenerate objective that collapses into repeating whatever
    # token that pattern favors most.
    input_ids = full_ids[:-1]
    targets = full_ids[1:]

    # -100 (IGNORE_INDEX) wherever the *target* is still a prompt token —
    # i.e. don't train the model to predict the prompt itself, only the
    # response (+ EOS). The last prompt-token position is a real target
    # (predicting the *first* response token, given the full prompt as
    # context) — that's why this checks (i + 1) >= n_prompt, not i >= n_prompt.
    labels = [tok if (i + 1) >= n_prompt else IGNORE_INDEX for i, tok in enumerate(targets)]

    return {"input_ids": input_ids, "labels": labels}


def collate(tokenizer, examples: list[dict], block_size: int, device: str):
    max_len = max(len(e["input_ids"]) for e in examples)
    max_len = min(max_len, block_size)

    input_ids, labels, attention_mask = [], [], []
    for e in examples:
        ids = e["input_ids"][:max_len]
        lbl = e["labels"][:max_len]
        pad_len = max_len - len(ids)

        input_ids.append(ids + [tokenizer.eos_id] * pad_len)  # pad value doesn't matter, it's masked out
        labels.append(lbl + [IGNORE_INDEX] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.bool, device=device),
    )


def tokenize_examples(tokenizer, raw_examples: list[dict], block_size: int) -> list[dict]:
    """Tokenize the full (prompt, response) list once. Call this before the
    training loop and reuse the result across every epoch/eval call —
    iterate_epoch() used to call build_example() fresh on every invocation
    (every epoch, and every eval call, ~13x per SFT run), needlessly
    re-tokenizing the same fixed dataset repeatedly."""
    return [build_example(tokenizer, e["prompt"], e["response"], block_size) for e in raw_examples]


def iterate_epoch(examples: list[dict], tokenizer, block_size: int, batch_size: int, device: str, shuffle: bool = True):
    """Yields one epoch's worth of (input_ids, labels, attention_mask) batches.
    `examples` must already be tokenized (see tokenize_examples) — this only
    shuffles/batches/pads, it doesn't re-tokenize."""
    if shuffle:
        examples = examples.copy()  # don't shuffle the caller's cached list in place
        random.shuffle(examples)

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        if not batch:
            continue
        yield collate(tokenizer, batch, block_size, device)
