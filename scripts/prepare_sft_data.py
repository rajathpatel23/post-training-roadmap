"""
Prepare SFT training data for Project 1.

YOUR JOB: implement the body of this script.

Input:  raw data in data/raw/
Output: data/processed/sft_train.jsonl
        data/eval/sft_eval.jsonl   <-- fix this split from day 1, never reshuffle

Expected output schema per line:
{
  "prompt":    "<system + user turn formatted for the model>",
  "response":  "<target output — must match the structured schema you define>"
}

Steps to implement:
1. Load raw dataset (HuggingFace datasets or local JSONL)
2. Filter / clean examples
3. Apply prompt template (see src/common/prompts.py)
4. Train/eval split — use a fixed seed=42
5. Write to JSONL

Run:
    python scripts/prepare_sft_data.py --config configs/sft/qwen05b_structured.yaml
"""

import argparse


def main(config_path: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
