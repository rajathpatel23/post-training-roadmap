"""All hyperparameters in one place. Override the common ones from the CLI, e.g.:
    python train.py --steps 12 --group-size 4 --lr 2e-4
"""
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # ---- model (tiny, from scratch; ~0.3M params) ----
    d_model: int = 128
    n_layer: int = 2
    n_head: int = 4
    d_ff: int = 256
    max_seq: int = 128
    dropout: float = 0.0

    # ---- task ----
    max_new_tokens: int = 48      # max completion length during rollouts

    # ---- GRPO ----
    prompts_per_step: int = 4     # B prompts per training step
    group_size: int = 4           # G completions sampled per prompt
    temperature: float = 1.0      # sampling temperature for rollouts
    top_k: int = 0                # 0 = disabled
    eps_clip: float = 0.2         # PPO-style clipping range
    kl_beta: float = 0.0          # KL-to-reference penalty weight; 0 = off
    mu_epochs: int = 1            # gradient epochs per batch of rollouts

    # ---- optimization ----
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ---- run ----
    steps: int = 200
    seed: int = 0
    eval_every: int = 50
    n_holdout: int = 200
    outdir: str = "runs/tiny"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tiny GRPO on arithmetic word problems")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-beta", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="runs/tiny")
    p.add_argument("--eval-every", type=int, default=50)
    a = p.parse_args(argv)
    cfg = Config(
        steps=a.steps, seed=a.seed, prompts_per_step=a.prompts_per_step,
        group_size=a.group_size, lr=a.lr, kl_beta=a.kl_beta,
        max_new_tokens=a.max_new_tokens, temperature=a.temperature,
        outdir=a.outdir, eval_every=a.eval_every,
    )
    return cfg
