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
import re

import torch

from bpe_data import BPETokenizer
from config import load_model_config
from model import TinyGPT

from src.common.generation import resolve_lm_device

BASE_DIR = os.path.dirname(__file__)
SFT_CHECKPOINT = os.path.join(BASE_DIR, "out_sft_bpe", "tinygpt_sft.pt")
SFT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "sft", "train.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "data", "dpo")

# out_sft_bpe/tinygpt_sft.pt was trained as this architecture. The weights
# file does not record the shape, so sampling reads the same yaml the
# trainer uses.
SFT_MODEL_CONFIG = "gpt_small_bpe"

K_SAMPLES = 6  # bumped from 4 — more draws helps once the heuristic below is more graduated too
MAX_NEW_TOKENS = 80

# Closed-class markers only. A bonus that grows with every "thou" is a
# verifier the policy can stuff. One distinct hit is the whole register
# credit. Overlap with the movie line is not a term: it would pay the
# model for copying modern wording, which is the opposite of the rewrite.
REGISTER_MARKERS = frozenset({
    "thee", "thou", "thy", "thine", "hath", "doth", "dost",
    "prithee", "wherefore", "whilst", "naught",
})
OBJECTIVE_KEYS = ("form", "register", "words")
_LEXICON: set[str] | None = None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _lexicon() -> set[str]:
    global _LEXICON
    if _LEXICON is None:
        from bpe_data import SHAKESPEARE_PATH

        with open(SHAKESPEARE_PATH, encoding="utf-8") as f:
            _LEXICON = set(_tokens(f.read()))
    return _LEXICON


def score_objectives(text: str, hit_eos: bool) -> dict[str, float]:
    """Form, register, and real-word fraction, kept as separate numbers.
    A single sum lets the widest-swinging check hide the others."""
    return {
        "form": _form_score(text, hit_eos),
        "register": 1.0 if REGISTER_MARKERS.intersection(_tokens(text)) else 0.0,
        "words": _known_word_fraction(text),
    }


def score_completion(text: str, hit_eos: bool) -> float:
    """One number for ranking a completion. DPO training does not call this.
    The gold pairs are already written. This ranks samples when a preference
    file is being built."""
    parts = score_objectives(text, hit_eos)
    return sum(parts[key] for key in OBJECTIVE_KEYS)


def _known_word_fraction(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    known = _lexicon()
    return sum(tok in known for tok in tokens) / len(tokens)


def _form_score(text: str, hit_eos: bool) -> float:
    """Higher = better shaped. Rule-based — no LLM judge needed.

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

    model_cfg = load_model_config(SFT_MODEL_CONFIG)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=model_cfg.block_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        dropout=model_cfg.dropout,
    ).to(device)
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
