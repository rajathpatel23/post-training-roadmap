"""Evaluate a trained checkpoint on held-out problems.

Greedy accuracy measures what the policy does deterministically; pass@K
measures what it *can* do when allowed K stochastic attempts -- the standard
capability metric in RLVR work, and the fairer one while the policy is still
exploring.

Usage:
    python eval.py --ckpt runs/tiny/model.pt               # greedy, full holdout
    python eval.py --ckpt runs/tiny/model.pt --n 40        # quick check
    python eval.py --ckpt runs/tiny/model.pt --samples 8   # adds pass@8
"""
import argparse
import torch

from tokenizer import Tokenizer
from model import TinyTransformer, generate
from data import make_pool
from rewards import score_completion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--samples", type=int, default=0,
                    help="if >0, also report pass@K with K stochastic samples")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ckpt["cfg"]
    tok = Tokenizer()
    model = TinyTransformer(len(tok), d_model=c["d_model"], n_layer=c["n_layer"],
                            n_head=c["n_head"], d_ff=c["d_ff"],
                            max_seq=c["max_seq"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    gen = torch.Generator().manual_seed(a.seed)

    problems = make_pool(seed=10_000 + c["seed"], n=a.n)
    correct, formatted = 0, 0
    passk = 0
    for i, p in enumerate(problems):
        prompt_ids = [tok.bos_id] + tok.encode(p.prompt)
        comp_ids, _ = generate(model, prompt_ids, c["max_new_tokens"],
                               temperature=0.0, eos_id=tok.eos_id)
        _, info = score_completion(comp_ids, tok, p.answer)
        correct += info["correct"]
        formatted += info["formatted"]
        if a.samples > 0:
            ok = False
            for _ in range(a.samples):
                s_ids, _ = generate(model, prompt_ids, c["max_new_tokens"],
                                    temperature=a.temperature, generator=gen,
                                    eos_id=tok.eos_id)
                _, s_info = score_completion(s_ids, tok, p.answer)
                ok = ok or s_info["correct"]
            passk += ok
        if i < a.show:
            print(f"Q: {p.prompt}\nA: {tok.decode(comp_ids)}\n"
                  f"   -> true={p.answer} correct={info['correct']}\n")
    print(f"greedy accuracy: {correct / a.n:.3f} ({correct}/{a.n}) | "
          f"formatted: {formatted / a.n:.3f}")
    if a.samples > 0:
        print(f"pass@{a.samples}: {passk / a.n:.3f} ({passk}/{a.n})")


if __name__ == "__main__":
    main()
