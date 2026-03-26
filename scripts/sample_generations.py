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

from typing import Any


import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.logging import init_run, log_samples, finish_run

def main(
    model_path: str,
    prompts_path: str,
    n: int,
    output_path: str,
    pretty: bool = False,
    *,
    max_new_tokens: int = 100,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    log_table_name: str = "sample_generations",
) -> None:
    run = None
    if wandb_project:
        run = init_run(
            {
                "model_path": model_path,
                "prompts_path": prompts_path,
                "n": n,
                "max_new_tokens": max_new_tokens,
            },
            output_dir=str(Path(output_path).parent),
            project=wandb_project,
            run_name=wandb_run_name,
            entity=wandb_entity,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to("mps")

    with open(prompts_path, "r") as f:
        prompts = [
            json.loads(line)["prompt"] for line in f if line.strip()
        ][:n]

    outputs = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,  # add "<|im_start|>assistant\n" so model starts generating
        )
        inputs = inputs.to("mps")
        output = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        outputs.append(output[0])

    records = [
        {"prompt": p, "output": tokenizer.decode(o, skip_special_tokens=True)}
        for p, o in zip(prompts, outputs)
    ]

    with open(output_path, "w") as f:
        if pretty:
            json.dump(records, f, indent=2)
        else:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if run is not None:
        # W&B table columns: prompt / base_output / trained_output.
        samples = [{"prompt": r["prompt"], "trained_output": r["output"]} for r in records]
        # step=0 so it’s stable and doesn’t depend on training global steps.
        log_samples(run, samples, step=0, table_name=log_table_name)
        finish_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON for human readability")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--wandb_project", default=None, help="Enable W&B logging if set")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--log_table_name", default="sample_generations")
    args = parser.parse_args()
    main(
        args.model,
        args.prompts,
        args.n,
        args.output,
        args.pretty,
        max_new_tokens=args.max_new_tokens,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        log_table_name=args.log_table_name,
    )
