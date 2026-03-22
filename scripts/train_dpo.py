"""
Train DPO model — Project 2.

YOUR JOB: implement the body of this script.

Key things to wire up:
1. Load config (YAML)
2. Load tokenizer and model — reference model (frozen) + policy model (LoRA)
3. Load preference dataset (prompt / chosen / rejected)
4. Instantiate trl.DPOTrainer with beta from config
5. Call trainer.train()
6. Log to W&B: pairwise win rate on held-out set, verbosity shift

Important DPO-specific concerns to handle:
- reference model must be frozen (no gradient updates)
- log_probs of chosen and rejected under reference model — understand what TRL does for you vs. what you must verify
- watch for verbosity bias in chosen responses pulling training in a bad direction

Run:
    python scripts/train_dpo.py --config configs/dpo/qwen05b_dpo.yaml
"""

import argparse


def main(config_path: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
