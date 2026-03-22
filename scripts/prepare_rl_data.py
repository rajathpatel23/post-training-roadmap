"""
Prepare prompts for Project 3 (GRPO / RLOO).

YOUR JOB: implement the body of this script.

Input:  verifiable task dataset (GSM8K subset, JSON transform task, etc.)
Output: data/processed/rl_train.jsonl
        data/eval/rl_eval.jsonl

Expected output schema per line:
{
  "prompt":          "<prompt text>",
  "ground_truth":    "<exact answer or expected output for verifier>"
}

Note: no "response" field — the model generates during RL training.
The verifier in src/rl/verifier.py checks against ground_truth.

Run:
    python scripts/prepare_rl_data.py --config configs/rl/qwen05b_grpo.yaml
"""

import argparse


def main(config_path: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
