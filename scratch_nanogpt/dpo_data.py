"""
Build DPO preference pairs: sample K completions per prompt from the SFT
model, score each with a cheap rule-based heuristic (no LLM judge, no human
labeling — same "verifiable reward" approach used elsewhere in this repo),
best-scored becomes "chosen," worst-scored becomes "rejected." Prompts
where all K completions tie are skipped — a preference pair needs an
actual preference, not a coin flip.

The heuristic targets the exact same behavior SFT was trained on ("one
complete, well-formed utterance that stops cleanly") — DPO's job is to
sharpen that further using the SFT model's own sample quality variance as
signal, not introduce a new objective.

Run:
    python dpo_data.py
"""

import json
import os
import random

import torch

from bpe_data import BPETokenizer
from model import TinyGPT

from src.common.generation import resolve_lm_device

BASE_DIR = os.path.dirname(__file__)
SFT_CHECKPOINT = os.path.join(BASE_DIR, "out_sft_bpe", "tinygpt_sft.pt")
SFT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "sft", "train.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "data", "dpo")

# Must match out_sft_bpe/'s architecture (gpt_small_bpe.yaml).
MODEL_CFG = dict(block_size=256, n_layer=12, n_head=8, n_embd=128, dropout=0.1)

K_SAMPLES = 6  # bumped from 4 — more draws helps once the heuristic below is more graduated too
MAX_NEW_TOKENS = 80


def score_completion(text: str, hit_eos: bool) -> float:
    """Higher = better. Rule-based — no LLM judge needed.

    First version of this scored almost everything the same: SFT already
    solved "stop cleanly + end in punctuation" almost universally (see
    reports/scratch_nanogpt.md's SFT Training section), so a mostly-binary
    heuristic built around that gave near-zero score separation among K
    samples per prompt — 284/287 prompts got skipped on the first
    generation pass. This version scores the quality axes SFT did NOT fully
    solve instead: rambling length, repetition (bigram + trigram, not just
    trigram), and run-on/comma-heavy structure — the actual unevenness
    visible between e.g. KING RICHARD III's clean output and JULIET's
    rambling one in the SFT qualitative comparison.
    """
    score = 1.0 if hit_eos else -2.0  # stopping cleanly still matters, just isn't the whole signal anymore

    stripped = text.strip()
    if not stripped:
        return score - 5.0  # fully degenerate

    if stripped[-1] in ".?!":
        score += 0.5
    else:
        score -= 1.0  # no terminal punctuation even after hitting EOS is still a bad sign

    words = stripped.split()
    n_words = len(words)

    if n_words < 2:
        score -= 2.0  # degenerate / near-empty
    elif n_words > 20:
        score -= 0.15 * (n_words - 20)  # graduated penalty for excessive rambling length

    # Bigram + trigram repetition (not trigram-only) — catches shorter loops
    # like "thou hast thou hast" that a trigram-only check would miss.
    for n in (2, 3):
        if n_words >= n + 1:
            ngrams = [tuple(words[i : i + n]) for i in range(n_words - n + 1)]
            n_unique = len(set(ngrams))
            if n_unique < len(ngrams):
                score -= 0.5 * (len(ngrams) - n_unique)  # graduated by how much repeats

    # Comma/semicolon density as a rough run-on-sentence proxy.
    punct_heavy = stripped.count(",") + stripped.count(";")
    if n_words > 0 and punct_heavy / n_words > 0.3:
        score -= 1.0

    return score


@torch.no_grad()
def sample_k(model, tokenizer, prompt: str, k: int, device: str, max_new_tokens: int = MAX_NEW_TOKENS):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    completions = []
    for _ in range(k):
        out_ids = model.generate(
            idx, max_new_tokens=max_new_tokens, temperature=0.9, top_k=40, eos_id=tokenizer.eos_id
        )[0].tolist()
        response_ids = out_ids[idx.shape[1] :]
        hit_eos = tokenizer.eos_id in response_ids
        if hit_eos:
            response_ids = response_ids[: response_ids.index(tokenizer.eos_id)]
        completions.append({"text": tokenizer.decode(response_ids), "hit_eos": hit_eos})
    return completions


def main():
    device = resolve_lm_device()
    tokenizer = BPETokenizer()

    model = TinyGPT(vocab_size=tokenizer.vocab_size, **MODEL_CFG).to(device)
    model.load_state_dict(torch.load(SFT_CHECKPOINT, map_location=device))
    print(f"loaded SFT checkpoint: {SFT_CHECKPOINT}")

    with open(SFT_TRAIN_PATH, encoding="utf-8") as f:
        prompts = sorted({json.loads(line)["prompt"] for line in f})
    print(f"{len(prompts)} unique prompts")

    pairs = []
    skipped_no_separation = 0
    for i, prompt in enumerate(prompts):
        completions = sample_k(model, tokenizer, prompt, K_SAMPLES, device)
        for c in completions:
            c["score"] = score_completion(c["text"], c["hit_eos"])
        completions.sort(key=lambda c: c["score"], reverse=True)
        best, worst = completions[0], completions[-1]
        if best["score"] == worst["score"]:
            skipped_no_separation += 1
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": best["text"],
                "rejected": worst["text"],
                "chosen_score": best["score"],
                "rejected_score": worst["score"],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts processed, {len(pairs)} pairs so far")

    print(
        f"built {len(pairs)} preference pairs "
        f"({skipped_no_separation} prompts skipped — no score separation among {K_SAMPLES} samples)"
    )

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

    print("\n--- 15 random pairs for manual inspection ---")
    for p in random.sample(pairs, min(15, len(pairs))):
        print(f"\nPrompt: {p['prompt']}")
        print(f"  chosen   (score={p['chosen_score']:+.1f}): {p['chosen']!r}")
        print(f"  rejected (score={p['rejected_score']:+.1f}): {p['rejected']!r}")


if __name__ == "__main__":
    main()
