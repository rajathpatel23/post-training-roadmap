"""GRPO training loop on arithmetic word problems.

Each step:
    1. draw B prompts from the training pool
    2. for each prompt, sample G completions and score them (rewards.py)
    3. group-normalize rewards into advantages (grpo.py)
    4. one GRPO policy update; log reward / KL / entropy / length

Checkpoints + a CSV log go to --outdir. Example:
    python train.py --steps 200
    python train.py --steps 12 --prompts-per-step 2 --group-size 4   # smoke test
"""
import csv
import os
import random
import time

import torch

from config import parse_args
from tokenizer import Tokenizer
from model import TinyTransformer, generate
from data import make_pool
from rewards import score_completion
from grpo import sample_group, group_advantages, grpo_update


def evaluate(model, tok, problems, max_new_tokens):
    """Greedy-decode each holdout problem; report exact-match accuracy."""
    correct, formatted = 0, 0
    for p in problems:
        comp_ids, _ = generate(
            model, [tok.bos_id] + tok.encode(p.prompt),
            max_new_tokens, temperature=0.0, eos_id=tok.eos_id)
        _, info = score_completion(comp_ids, tok, p.answer)
        correct += info["correct"]
        formatted += info["formatted"]
    n = len(problems)
    return correct / n, formatted / n


def main():
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 1)   # sampling stream

    tok = Tokenizer()
    model = TinyTransformer(len(tok), d_model=cfg.d_model, n_layer=cfg.n_layer,
                            n_head=cfg.n_head, d_ff=cfg.d_ff,
                            max_seq=cfg.max_seq, dropout=cfg.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} | vocab: {len(tok)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    ref_model = None
    if cfg.kl_beta > 0:                       # frozen reference for the KL penalty
        import copy
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    train_pool = make_pool(seed=cfg.seed, n=2000)
    holdout = make_pool(seed=10_000 + cfg.seed, n=cfg.n_holdout)

    log_path = os.path.join(cfg.outdir, "log.csv")
    logf = open(log_path, "w", newline="")
    writer = csv.DictWriter(logf, fieldnames=[
        "step", "reward", "kl", "entropy", "mean_len", "loss",
        "holdout_acc", "secs"])
    writer.writeheader()

    t_start = time.time()
    for step in range(cfg.steps):
        t0 = time.time()
        prompts = rng.sample(train_pool, cfg.prompts_per_step)

        rollouts, advantages, rewards = [], [], []
        for p in prompts:
            group = sample_group(model, tok, p.prompt, cfg, gen)
            r = [score_completion(g.comp_ids, tok, p.answer)[0] for g in group]
            rollouts.extend(group)
            advantages.extend(group_advantages(r))
            rewards.extend(r)

        m = grpo_update(model, ref_model, opt, rollouts, advantages, cfg)
        mean_r = sum(rewards) / len(rewards)

        holdout_acc = ""
        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            acc, fmt = evaluate(model, tok, holdout, cfg.max_new_tokens)
            holdout_acc = f"{acc:.3f}"
            print(f"  [eval] holdout acc={acc:.3f} formatted={fmt:.3f}")

        secs = time.time() - t0
        writer.writerow({"step": step, "reward": f"{mean_r:.4f}",
                         "kl": f"{m['kl']:.5f}", "entropy": f"{m['entropy']:.4f}",
                         "mean_len": f"{m['mean_len']:.1f}",
                         "loss": f"{m['loss']:.4f}",
                         "holdout_acc": holdout_acc, "secs": f"{secs:.1f}"})
        logf.flush()
        print(f"step {step:4d} | reward {mean_r:.3f} | kl {m['kl']:.4f} | "
              f"ent {m['entropy']:.3f} | len {m['mean_len']:.1f} | "
              f"loss {m['loss']:.4f} | {secs:.1f}s")

    ckpt = os.path.join(cfg.outdir, "model.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, ckpt)
    print(f"saved {ckpt} | total {time.time() - t_start:.0f}s | log {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
