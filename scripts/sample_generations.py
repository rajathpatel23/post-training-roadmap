"""
Run greedy decoding on N prompts and save outputs to file.

Use this for Day 2: load base model, run inference on 20 prompts, save to file.
Also useful for qualitative comparison between checkpoints.

YOUR JOB: implement the body of this script.

Run:
    # Day 2 baseline
    python scripts/sample_generations.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --prompts data/eval/sft_eval.jsonl \
        --n 20 \
        --output reports/baseline_samples.jsonl

    # After training
    python scripts/sample_generations.py \
        --model outputs/project1_sft/checkpoint-best \
        --prompts data/eval/sft_eval.jsonl \
        --n 20 \
        --output reports/project1_samples.jsonl
"""

import argparse


def main(model_path: str, prompts_path: str, n: int, output_path: str) -> None:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.model, args.prompts, args.n, args.output)
