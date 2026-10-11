"""Evaluate a trained checkpoint on held-out problems.

Greedy / pass@1: what the policy prefers.
pass@K: at least one of K samples is correct — the gold-oracle upper bound
        (same as “best-of-K” when the ranker is the checker).
majority@K: most common extracted answer among K samples. No gold at pick time.
format-best@K: pick the sample with the strongest format rungs, then score it.
        A gold-blind ranker. Usually sits between greedy and pass@K.

Usage:
    python eval.py --ckpt runs/tiny/model.pt
    python eval.py --ckpt runs/tiny/model.pt --n 40
    python eval.py --ckpt runs/tiny/model.pt --n 40 --samples 8
    python eval.py --ckpt runs/tiny/model.pt --n 40 --samples 8 --max-new-tokens 24
"""
import argparse
from collections import Counter

import torch

from tokenizer import Tokenizer
from model import TinyTransformer, generate
from data import make_pool
from rewards import extract_answer, score_completion


def majority_pred(preds: list[int | None]) -> int | None:
    vals = [p for p in preds if p is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def format_rank(info: dict) -> int:
    # rung2 implies the full "answer :" pattern; rung1 is the word only.
    return int(info["rung1"]) + 2 * int(info["formatted"])


def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["cfg"]
    tok = Tokenizer()
    model = TinyTransformer(
        len(tok),
        d_model=c["d_model"],
        n_layer=c["n_layer"],
        n_head=c["n_head"],
        d_ff=c["d_ff"],
        max_seq=c["max_seq"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--samples", type=int, default=0,
                    help="if >0, also report pass@K / majority@K / format-best@K")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override checkpoint max_new_tokens (length sweep)")
    a = ap.parse_args()

    model, tok, c = load_model(a.ckpt)
    max_new = a.max_new_tokens if a.max_new_tokens is not None else c["max_new_tokens"]
    gen = torch.Generator().manual_seed(a.seed)
    problems = make_pool(seed=10_000 + c["seed"], n=a.n)

    greedy_ok = formatted = 0
    passk = majk = fmtk = 0

    for i, p in enumerate(problems):
        prompt_ids = [tok.bos_id] + tok.encode(p.prompt)
        greedy_ids, _ = generate(model, prompt_ids, max_new,
                                 temperature=0.0, eos_id=tok.eos_id)
        _, g_info = score_completion(greedy_ids, tok, p.answer)
        greedy_ok += g_info["correct"]
        formatted += g_info["formatted"]

        if a.samples > 0:
            preds, infos = [], []
            for _ in range(a.samples):
                s_ids, _ = generate(model, prompt_ids, max_new,
                                    temperature=a.temperature, generator=gen,
                                    eos_id=tok.eos_id)
                _, s_info = score_completion(s_ids, tok, p.answer)
                preds.append(extract_answer(s_ids, tok))
                infos.append(s_info)
            passk += any(info["correct"] for info in infos)
            majk += majority_pred(preds) == p.answer
            fmtk += infos[max(range(a.samples), key=lambda j: format_rank(infos[j]))]["correct"]

        if i < a.show:
            print(f"Q: {p.prompt}\nA: {tok.decode(greedy_ids)}\n"
                  f"   -> true={p.answer} greedy={g_info['correct']}\n")

    n = a.n
    print(f"greedy / pass@1: {greedy_ok / n:.3f} ({greedy_ok}/{n}) | "
          f"formatted: {formatted / n:.3f}")
    if a.samples > 0:
        k = a.samples
        print(f"majority@{k}:    {majk / n:.3f} ({majk}/{n})   "
              f"(mode of extracted answers; no gold at pick time)")
        print(f"format-best@{k}: {fmtk / n:.3f} ({fmtk}/{n})   "
              f"(pick strongest format rungs, then score)")
        print(f"pass@{k}:        {passk / n:.3f} ({passk}/{n})   "
              f"(oracle / gold picker — same as best-of-{k} with this checker)")
        print(f"max_new_tokens={max_new}  temperature={a.temperature}")
        print("read: greedy ≤ majority ≈ format-best ≤ pass@k. "
              "GRPO’s job is to lift greedy toward pass@k.")


if __name__ == "__main__":
    main()
