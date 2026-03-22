"""
Prepare preference pairs for Project 2 (DPO).

YOUR JOB: implement the body of this script.

Input:  raw preference dataset (Ultrafeedback or Anthropic HH)
Output: data/processed/pref_train.jsonl
        data/eval/pref_eval.jsonl

Expected output schema per line (TRL DPOTrainer format):
{
  "prompt":   "<prompt>",
  "chosen":   "<preferred response>",
  "rejected": "<dispreferred response>"
}

Before running training:
- Manually inspect 50 pairs. Tag: noise, verbosity bias, weak-preference.
- Write findings in notes/project2_data_inspection.md

Run:
    python scripts/prepare_pref_data.py --config configs/dpo/qwen05b_dpo.yaml
"""

import argparse


def main(config_path: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
