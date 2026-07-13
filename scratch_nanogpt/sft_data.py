"""
Build the SFT dataset: prompt = speaker tag, response = one complete,
well-formed sentence of that speaker's next line.

Directly targets the failure mode observed in Run 1 (pretraining-only
generation cuts off mid-word/mid-clause, no natural stopping point) —
teaches the base model "given a speaker, produce one complete utterance
and stop," using only data already in tiny-shakespeare.txt (no new
sourcing, no tokenizer-vocab risk since it's the same corpus).

Raw corpus structure: blocks separated by blank lines, first line of each
block is a speaker tag ("MENENIUS:"), remaining lines are that speech
word-wrapped across multiple raw lines — NOT one sentence per raw line.
So we rejoin wrapped lines before splitting into sentences.

Run:
    python sft_data.py
"""

import json
import os
import re
import random

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "sft")

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")
MIN_RESPONSE_LEN = 8
MAX_RESPONSE_LEN = 150


def is_speaker_tag(line: str) -> bool:
    line = line.strip()
    if not line.endswith(":"):
        return False
    if len(line) > 30:
        return False
    # Reject lines with sentence punctuation before the colon — those are
    # dialogue that happens to contain a colon, not a speaker tag.
    body = line[:-1]
    return not any(p in body for p in ".?!")


def first_complete_sentence(text: str) -> str | None:
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    for s in sentences:
        s = s.strip()
        if MIN_RESPONSE_LEN <= len(s) <= MAX_RESPONSE_LEN and s[-1] in ".?!":
            return s
    return None


def build_pairs() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    pairs = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        speaker_line, dialogue_lines = lines[0].strip(), lines[1:]
        if not is_speaker_tag(speaker_line):
            continue

        # Rejoin word-wrapped raw lines into one continuous speech.
        joined = " ".join(line.strip() for line in dialogue_lines if line.strip())
        response = first_complete_sentence(joined)
        if response is None:
            continue

        pairs.append({"prompt": speaker_line, "response": response})

    return pairs


def main():
    pairs = build_pairs()
    print(f"built {len(pairs)} (prompt, response) pairs from {DATA_PATH}")

    random.seed(1337)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(OUT_DIR, "eval.jsonl"), "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"train: {len(train_pairs)}  eval: {len(val_pairs)}  -> {OUT_DIR}/")

    print("\n--- 15 random examples for manual inspection ---")
    for p in random.sample(pairs, min(15, len(pairs))):
        print(f"  {p['prompt']:<20} -> {p['response']}")


if __name__ == "__main__":
    main()
