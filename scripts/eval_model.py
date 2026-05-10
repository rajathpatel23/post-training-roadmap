"""
Run evaluation on a checkpoint against the fixed eval set.

Scores **assistant completions only** (tokens after the chat prompt), so JSON
metrics are not polluted by system/user prompt text.

Output (write to reports/ and log to W&B):
  - checkpoint name, eval path, generation settings
  - exact metric scores (format validity, task success, etc.)
  - sampled generations (base vs trained)
  - grouped failure buckets (JSON validity comparison)

Run:
    python scripts/eval_model.py \
        --config configs/sft/qwen05b_structured.yaml \
        --checkpoint outputs/project1_sft/checkpoint-best \
        --eval_path data/eval/sft_eval.jsonl \
        --output_dir reports/
"""

import argparse


import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.config import ExperimentConfig
from src.common.generation import decode_assistant_completion, resolve_lm_device
from src.common.io import read_jsonl, write_jsonl
from src.common.logging import init_run, log_metrics, log_samples, finish_run
from src.evals.exact_match import (
    avg_response_length,
    exact_field_presence_rate,
    format_validity_rate,
    is_parseable_json,
)
from src.evals.qualitative_dump import bucket_failures


def main(config_path: str, checkpoint: str, eval_path: str, output_dir: str) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts (fixed order, no shuffling).
    records = read_jsonl(eval_path)
    prompts = [r["prompt"] for r in records if "prompt" in r]
    if not prompts:
        raise ValueError(f"No `prompt` fields found in {eval_path}")

    checkpoint_name = os.path.basename(os.path.normpath(checkpoint))

    device = resolve_lm_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.primary)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_with(model_path: str) -> list[str]:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        model.to(device)

        outs: list[str] = []
        with torch.no_grad():
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                inputs = inputs.to(device)

                input_ids = inputs["input_ids"]
                gen_out = model.generate(
                    **inputs,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=cfg.eval.generation_max_new_tokens,
                    do_sample=cfg.eval.generation_do_sample,
                    temperature=cfg.eval.generation_temperature,
                )
                outs.append(
                    decode_assistant_completion(tokenizer, input_ids, gen_out),
                )

        del model
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass
        return outs

    base_outputs = generate_with(cfg.model.primary)
    trained_outputs = generate_with(checkpoint)

    # Metrics
    metrics: dict[str, float] = {
        "eval/format_validity_rate": float(format_validity_rate(trained_outputs)),
        "eval/avg_response_length": float(avg_response_length(trained_outputs)),
    }
    if cfg.eval.required_fields:
        metrics["eval/exact_field_presence_rate"] = float(
            exact_field_presence_rate(trained_outputs, cfg.eval.required_fields)
        )

    # Optional task success if the dataset includes ground_truth fields.
    if all("ground_truth" in r for r in records):
        ground_truths = [str(r["ground_truth"]) for r in records]

        def _verifier_exact(gen: str, gt: str) -> float:
            return 1.0 if gen.strip() == gt.strip() else 0.0

        success = sum(_verifier_exact(g, gt) for g, gt in zip(trained_outputs, ground_truths)) / len(ground_truths)
        metrics["eval/task_success_rate_exact"] = float(success)

    # Sample side-by-side (qualitative + failure buckets).
    n_samples = min(cfg.eval.num_sample_generations, len(prompts))

    def _is_json_valid(s: str) -> bool:
        return is_parseable_json(s)

    base_valid = [_is_json_valid(o) for o in base_outputs[:n_samples]]
    trained_valid = [_is_json_valid(o) for o in trained_outputs[:n_samples]]

    failure_labels: list[str] = []
    for bv, tv in zip(base_valid, trained_valid, strict=True):
        if bv and tv:
            failure_labels.append("both_valid_json")
        elif bv and not tv:
            failure_labels.append("base_valid_trained_invalid_json")
        elif not bv and tv:
            failure_labels.append("base_invalid_trained_valid_json")
        else:
            failure_labels.append("both_invalid_json")

    buckets = bucket_failures(generations=trained_outputs[:n_samples], labels=failure_labels)

    samples = []
    for i in range(n_samples):
        samples.append(
            {
                "prompt": prompts[i],
                "base_output": base_outputs[i],
                "trained_output": trained_outputs[i],
            }
        )

    # Write artifacts (reports/ by convention).
    metrics_path = os.path.join(output_dir, f"metrics_{checkpoint_name}.json")
    metrics_payload = {
        "checkpoint": checkpoint_name,
        "eval_path": eval_path,
        "base_model": cfg.model.primary,
        "device": device,
        "metrics": metrics,
        "eval_config": {
            "generation_max_new_tokens": cfg.eval.generation_max_new_tokens,
            "generation_do_sample": cfg.eval.generation_do_sample,
            "generation_temperature": cfg.eval.generation_temperature,
            "num_sample_generations": cfg.eval.num_sample_generations,
            "required_fields": cfg.eval.required_fields,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json as _json

        _json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    samples_path = os.path.join(output_dir, f"samples_{checkpoint_name}.jsonl")
    write_jsonl(samples, samples_path)

    buckets_path = os.path.join(output_dir, f"failure_buckets_{checkpoint_name}.json")
    with open(buckets_path, "w", encoding="utf-8") as f:
        import json as _json

        _json.dump(buckets, f, indent=2, ensure_ascii=False)

    # W&B logging
    run = init_run(
        {
            "project": cfg.project,
            "run_name": cfg.run_name,
            "model_primary": cfg.model.primary,
            "checkpoint": checkpoint,
            "eval_path": eval_path,
        },
        output_dir=output_dir,
        project=os.environ.get("WANDB_PROJECT", cfg.project),
        run_name=f"{cfg.run_name}-eval-{checkpoint_name}",
    )

    log_metrics(run, metrics, step=0)
    log_samples(run, samples, step=0, table_name="eval_samples")
    finish_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.eval_path, args.output_dir)
