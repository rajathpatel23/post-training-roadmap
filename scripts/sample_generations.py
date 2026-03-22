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

from transformers import AutoModelForCausalLM, AutoTokenizer

def main(
    model_path: str, prompts_path: str, n: int, output_path: str, pretty: bool = False
) -> None:
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
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        output = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=100)
        outputs.append(output[0])

    records = [
        {"prompt": p, "output": tokenizer.decode(o, skip_special_tokens=True)}
        for p, o in zip[tuple[Any, Any]](prompts, outputs)
    ]

    with open(output_path, "w") as f:
        if pretty:
            json.dump(records, f, indent=2)
        else:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON for human readability")
    args = parser.parse_args()
    main(args.model, args.prompts, args.n, args.output, args.pretty)
