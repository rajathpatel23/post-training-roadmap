"""
Run evaluation on a checkpoint against the fixed eval set.

YOUR JOB: implement the body of this script.

Output (write to reports/ and log to W&B):
  - checkpoint name
  - dataset split
  - exact metric scores (format validity, task success, etc.)
  - 20 sampled generations (base model vs. trained)
  - grouped failure categories
  - generation config used

Run:
    python scripts/eval_model.py \
        --config configs/sft/qwen05b_structured.yaml \
        --checkpoint outputs/project1_sft/checkpoint-best \
        --eval_path data/eval/sft_eval.jsonl \
        --output_dir reports/
"""

import argparse


def main(config_path: str, checkpoint: str, eval_path: str, output_dir: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.eval_path, args.output_dir)
