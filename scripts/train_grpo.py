"""
Train GRPO model — Project 3.

YOUR JOB: implement the body of this script.

Key things to wire up:
1. Load config (YAML)
2. Load tokenizer and model
3. Load prompts + ground_truth from JSONL
4. Define reward function — calls src/rl/verifier.py
5. Instantiate trl.GRPOTrainer (or GRPOConfig)
6. Call trainer.train()
7. Log to W&B: verifier pass rate, avg reward, invalid generation rate

GRPO-specific things to understand before implementing:
- GRPO samples G completions per prompt, computes group-mean-normalized reward
- No separate reward model — reward comes from verifier
- KL penalty keeps policy close to reference
- Read the GRPO paper (or DeepSeek-R1 appendix) before writing this script

Run:
    python scripts/train_grpo.py --config configs/rl/qwen05b_grpo.yaml
"""

import argparse


def main(config_path: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
