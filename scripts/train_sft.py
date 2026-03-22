"""
Train SFT model — Project 1.

YOUR JOB: implement the body of this script.

Key things to wire up:
1. Load config (YAML) and merge with base.yaml
2. Load tokenizer and model (with LoRA via peft)
3. Load train/eval datasets from JSONL
4. Instantiate trl.SFTTrainer with:
   - TrainingArguments (from config)
   - LoraConfig (from config)
   - train_dataset, eval_dataset
5. Call trainer.train()
6. Save final checkpoint + adapter weights
7. Log to W&B

All hyperparams must come from the config file — no hardcoding.

Run:
    python scripts/train_sft.py --config configs/sft/qwen05b_structured.yaml
"""

import argparse


def main(config_path: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
