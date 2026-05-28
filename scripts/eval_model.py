"""
Run evaluation on a checkpoint against the fixed eval set.

Scores **assistant completions only** (tokens after the chat prompt), so JSON
metrics are not polluted by system/user prompt text.

Output (write to reports/ and log to W&B):
  - checkpoint name, eval path, generation settings
  - exact metric scores (format validity, entity micro P/R/F1, task success, etc.)
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

from src.common.config import ExperimentConfig
from src.common.generation import resolve_lm_device
from src.common.io import read_jsonl, write_jsonl
from src.common.logging import init_run, log_metrics, log_samples, finish_run
from src.common.model_loading import load_causal_lm_for_inference, load_tokenizer, release_model
from src.evals.generative_eval import generate_completions, score_generations
from src.evals.exact_match import is_parseable_json
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
    fp16 = cfg.training.fp16
    bf16 = cfg.training.bf16

    def generate_with(model_path: str) -> list[str]:
        tokenizer = load_tokenizer(model_path, base_model=cfg.model.primary)
        model = load_causal_lm_for_inference(
            model_path,
            base_model=cfg.model.primary,
            device=device,
            fp16=fp16,
            bf16=bf16,
        )
        outs = generate_completions(
            model,
            tokenizer,
            prompts,
            device=device,
            max_new_tokens=cfg.eval.generation_max_new_tokens,
            do_sample=cfg.eval.generation_do_sample,
            temperature=cfg.eval.generation_temperature,
        )
        release_model(model)
        return outs

    base_outputs = generate_with(cfg.model.primary)
    trained_outputs = generate_with(checkpoint)

    ground_truths = [str(r["ground_truth"]) for r in records] if all("ground_truth" in r for r in records) else None
    required_fields = cfg.eval.required_fields or []

    trained_metrics = score_generations(trained_outputs, ground_truths, required_fields=required_fields)
    base_metrics = score_generations(base_outputs, ground_truths, required_fields=required_fields)

    metrics: dict[str, float] = {
        "eval/format_validity_rate": trained_metrics["format_validity_rate"],
        "eval/avg_response_length": trained_metrics["avg_response_length"],
    }
    if "exact_field_presence_rate" in trained_metrics:
        metrics["eval/exact_field_presence_rate"] = trained_metrics["exact_field_presence_rate"]
    if ground_truths is not None:
        metrics["eval/task_success_rate"] = trained_metrics["task_success_rate"]
        metrics["eval/entity_precision"] = trained_metrics["entity_precision"]
        metrics["eval/entity_recall"] = trained_metrics["entity_recall"]
        metrics["eval/entity_f1"] = trained_metrics["entity_f1"]
        metrics["eval/base_entity_precision"] = base_metrics["entity_precision"]
        metrics["eval/base_entity_recall"] = base_metrics["entity_recall"]
        metrics["eval/base_entity_f1"] = base_metrics["entity_f1"]

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
