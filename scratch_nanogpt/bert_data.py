"""
Data pipeline for TinyBERT: NSP sentence pairs + MLM masking, char-level.

"Sentences" = non-empty lines of tiny-shakespeare (it's dialogue, so each line
is already a natural short segment — no real sentence splitter needed).

Layout per example: [CLS] segA [SEP] segB [SEP], padded to block_size.
- NSP: 50% segB is the actual next line (label 0 = IsNext), 50% segB is a
  random line from elsewhere in the corpus (label 1 = NotNext) — same recipe
  as the original BERT paper.
- MLM: 15% of non-special, non-pad positions are selected; of those, 80%
  become [MASK], 10% a random vocab token, 10% left unchanged. Loss is
  computed only at selected positions (target -100 elsewhere = ignored by
  cross_entropy).
"""

import os
import random

import torch

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
PAD_ID, CLS_ID, SEP_ID, MASK_ID = range(4)  # reserved as the first 4 vocab ids


class CharTokenizerWithSpecials:
    """Same char vocab as data.py's CharTokenizer, but with 4 reserved special
    ids prepended so [PAD]/[CLS]/[SEP]/[MASK] never collide with real chars."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        offset = len(SPECIAL_TOKENS)
        for i, ch in enumerate(chars):
            self.stoi[ch] = i + offset
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)


def load_lines(val_fraction: float = 0.1):
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    lines = [line for line in text.split("\n") if line.strip()]

    tokenizer = CharTokenizerWithSpecials(text)

    n = int(len(lines) * (1 - val_fraction))
    return tokenizer, lines[:n], lines[n:]


def make_nsp_pair(lines: list[str], idx: int) -> tuple[str, str, int]:
    """Returns (seg_a, seg_b, label). label: 0 = IsNext, 1 = NotNext."""
    seg_a = lines[idx]
    if random.random() < 0.5 and idx + 1 < len(lines):
        seg_b = lines[idx + 1]
        label = 0
    else:
        rand_idx = random.randrange(len(lines))
        seg_b = lines[rand_idx]
        label = 1
    return seg_a, seg_b, label


def apply_mlm_masking(token_ids: list[int], vocab_size: int, mlm_probability: float) -> tuple[list[int], list[int]]:
    """Returns (masked_input_ids, mlm_labels). Only touches positions that
    aren't a special token (PAD/CLS/SEP already excluded by the caller)."""
    input_ids = list(token_ids)
    labels = [-100] * len(token_ids)

    for i, tok in enumerate(token_ids):
        if tok in (PAD_ID, CLS_ID, SEP_ID):
            continue
        if random.random() >= mlm_probability:
            continue

        labels[i] = tok  # predict the original token at this position
        r = random.random()
        if r < 0.8:
            input_ids[i] = MASK_ID
        elif r < 0.9:
            input_ids[i] = random.randrange(len(SPECIAL_TOKENS), vocab_size)
        # else: leave unchanged (10%) — still a loss target, model must not
        # assume "unmasked = definitely correct."

    return input_ids, labels


def build_example(tokenizer: CharTokenizerWithSpecials, seg_a: str, seg_b: str, nsp_label: int, block_size: int, mlm_probability: float):
    a_ids = tokenizer.encode(seg_a)
    b_ids = tokenizer.encode(seg_b)

    # Reserve 3 special tokens ([CLS] ... [SEP] ... [SEP]); truncate segments to fit.
    max_content = block_size - 3
    # Split remaining budget evenly between the two segments.
    a_budget = max_content // 2
    b_budget = max_content - a_budget
    a_ids = a_ids[:a_budget]
    b_ids = b_ids[:b_budget]

    token_ids = [CLS_ID] + a_ids + [SEP_ID] + b_ids + [SEP_ID]
    segment_ids = [0] * (1 + len(a_ids) + 1) + [1] * (len(b_ids) + 1)

    pad_len = block_size - len(token_ids)
    attention_mask = [1] * len(token_ids) + [0] * pad_len
    token_ids = token_ids + [PAD_ID] * pad_len
    segment_ids = segment_ids + [0] * pad_len

    masked_ids, mlm_labels = apply_mlm_masking(token_ids, tokenizer.vocab_size, mlm_probability)
    # Don't let masking touch padding positions (apply_mlm_masking already
    # skips CLS/SEP but PAD isn't special-cased there since it's just id 0 —
    # zero out any accidental loss on padding to be safe).
    for i in range(len(masked_ids)):
        if attention_mask[i] == 0:
            mlm_labels[i] = -100

    return {
        "input_ids": masked_ids,
        "segment_ids": segment_ids,
        "attention_mask": attention_mask,
        "mlm_labels": mlm_labels,
        "nsp_label": nsp_label,
    }


def get_batch(tokenizer, lines: list[str], block_size: int, batch_size: int, mlm_probability: float, device: str):
    examples = []
    for _ in range(batch_size):
        idx = random.randrange(len(lines))
        seg_a, seg_b, nsp_label = make_nsp_pair(lines, idx)
        examples.append(build_example(tokenizer, seg_a, seg_b, nsp_label, block_size, mlm_probability))

    input_ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long, device=device)
    segment_ids = torch.tensor([e["segment_ids"] for e in examples], dtype=torch.long, device=device)
    attention_mask = torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long, device=device)
    mlm_labels = torch.tensor([e["mlm_labels"] for e in examples], dtype=torch.long, device=device)
    nsp_labels = torch.tensor([e["nsp_label"] for e in examples], dtype=torch.long, device=device)

    return input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels
