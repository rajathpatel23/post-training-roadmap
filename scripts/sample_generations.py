"""
Run fixed-parameter decoding on N prompts and save outputs to file.

Each line stores the **assistant completion only** (decoded new tokens), not the
full prompt, so downstream JSON metrics match what `eval_model.py` scores.

Use for Day 2 baselines and qualitative checkpoint comparison.

Run:
    # Day 2 baseline
    python scripts/sample_generations.py \
        --config configs/sft/qwen05b_structured.yaml \
        --prompts data/eval/sft_eval.jsonl \
        --n 20 \
        --output reports/baseline_samples.jsonl \
        --metrics_output reports/baseline_metrics.json

    # After training
    python scripts/sample_generations.py \
        --model outputs/project1_sft/checkpoint-best \
        --config configs/sft/qwen05b_structured.yaml \
        --prompts data/eval/sft_eval.jsonl \
        --n 20 \
        --output reports/project1_samples.jsonl
"""

import argparse
import json
from pathlib import Path

from src.common.config import ExperimentConfig
from src.common.generation import resolve_lm_device
from src.common.io import read_jsonl, write_jsonl
from src.common.logging import init_run, log_samples, finish_run
from src.common.model_loading import load_causal_lm_for_inference, load_tokenizer, release_model
from src.evals.generative_eval import generate_completions, score_generations


def main(
    model_path: str | None,
    prompts_path: str,
    n: int,
    output_path: str,
    pretty: bool = False,
    *,
    config_path: str | None = None,
    max_new_tokens: int | None = None,
    do_sample: bool | None = None,
    temperature: float | None = None,
    metrics_output_path: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    log_table_name: str = "sample_generations",
) -> None:
    cfg = ExperimentConfig.from_yaml(config_path) if config_path else None
    if model_path is None:
        if cfg is None:
            raise ValueError("Either --model or --config must be provided")
        model_path = cfg.model.primary

    if max_new_tokens is None:
        max_new_tokens = cfg.eval.generation_max_new_tokens if cfg else 256
    if do_sample is None:
        do_sample = cfg.eval.generation_do_sample if cfg else False
    if temperature is None:
        temperature = cfg.eval.generation_temperature if cfg else 0.0
    required_fields = cfg.eval.required_fields if cfg else None

    run = None
    if wandb_project:
        run = init_run(
            {
                "model_path": model_path,
                "prompts_path": prompts_path,
                "n": n,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
            },
            output_dir=str(Path(output_path).parent),
            project=wandb_project,
            run_name=wandb_run_name,
            entity=wandb_entity,
        )

    device = resolve_lm_device()
    fp16 = cfg.training.fp16 if cfg else False
    bf16 = cfg.training.bf16 if cfg else False
    base_model = cfg.model.primary if cfg else None

    tokenizer = load_tokenizer(model_path, base_model=base_model)
    model = load_causal_lm_for_inference(
        model_path,
        base_model=base_model,
        device=device,
        fp16=fp16,
        bf16=bf16,
    )

    prompt_records = read_jsonl(prompts_path)[:n]
    prompts = [record["prompt"] for record in prompt_records]

    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )
    release_model(model)

    records: list[dict[str, str]] = [
        {"prompt": prompt, "output": text} for prompt, text in zip(prompts, outputs, strict=True)
    ]

    if pretty:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    else:
        write_jsonl(records, output_path)

    ground_truths = [record["ground_truth"] for record in prompt_records if "ground_truth" in record]
    gt_for_score = ground_truths if len(ground_truths) == len(outputs) else None
    metrics = score_generations(outputs, gt_for_score, required_fields=required_fields or None)

    if metrics_output_path:
        metrics_payload = {
            "model": model_path,
            "prompts_path": prompts_path,
            "output_path": output_path,
            "num_prompts": len(records),
            "generation_params": {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
            },
            "required_fields": required_fields or [],
            "metrics": metrics,
        }
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    if run is not None:
        # W&B table columns: prompt / base_output / trained_output.
        samples = [{"prompt": r["prompt"], "trained_output": r["output"]} for r in records]
        # step=0 so it’s stable and doesn’t depend on training global steps.
        log_samples(run, samples, step=0, table_name=log_table_name)
        finish_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON for human readability")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--metrics_output", default=None)
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
        config_path=args.config,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        metrics_output_path=args.metrics_output,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        log_table_name=args.log_table_name,
    )
