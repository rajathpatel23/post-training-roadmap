"""
Train SFT model — Project 1 (TRL `SFTTrainer` + optional LoRA).

JSONL rows use string `prompt` / `response` (or eval `ground_truth`). This script
maps them to TRL's *conversational* prompt-completion format so tokenization
uses `tokenizer.apply_chat_template` — aligned with `eval_model.py`.

Run from repo root:
    python scripts/train_sft.py --config configs/sft/qwen05b_structured.yaml

Smoke test (skips eval loop for speed; still tokenizes full train set once):
    python scripts/train_sft.py --config configs/sft/qwen05b_structured.yaml --max_steps 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.common.config import ExperimentConfig
from src.common.io import read_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


def _abs_data_path(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _rows_to_conv_sft(rows: list[dict], *, response_key: str) -> list[dict]:
    out: list[dict] = []
    for i, row in enumerate(rows):
        if "prompt" not in row:
            raise KeyError(f"Missing 'prompt' at row {i}")
        if response_key not in row:
            raise KeyError(f"Missing '{response_key}' at row {i}")
        prompt = row["prompt"]
        response = row[response_key]
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError(f"Row {i}: expected string prompt/{response_key}")
        out.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": response}],
            }
        )
    return out


def _build_sft_config(cfg: ExperimentConfig, *, max_steps: int | None, do_eval: bool) -> SFTConfig:
    report_to = cfg.logging.report_to
    if isinstance(report_to, str) and report_to.lower() == "none":
        report_to = "none"

    kwargs: dict = {
        "output_dir": _abs_data_path(cfg.training.output_dir),
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.per_device_train_batch_size,
        "num_train_epochs": cfg.training.num_train_epochs,
        "learning_rate": cfg.training.learning_rate,
        "gradient_accumulation_steps": cfg.training.grad_accumulation_steps,
        "warmup_ratio": cfg.training.warmup_ratio,
        "weight_decay": cfg.training.weight_decay,
        "lr_scheduler_type": cfg.training.lr_scheduler,
        "fp16": cfg.training.fp16,
        "bf16": cfg.training.bf16,
        "logging_steps": cfg.logging.logging_steps,
        "report_to": report_to,
        "seed": cfg.training.seed,
        "data_seed": cfg.training.seed,
        "dataloader_num_workers": cfg.training.dataloader_num_workers,
        "dataloader_pin_memory": False,
        "save_strategy": "steps",
        "save_steps": cfg.eval.save_steps,
        "save_total_limit": cfg.eval.save_total_limit,
        "load_best_model_at_end": cfg.eval.load_best_model_at_end if do_eval else False,
        "metric_for_best_model": cfg.eval.metric_for_best_model,
        "greater_is_better": False,
        "max_length": cfg.training.max_length,
        "completion_only_loss": True,
        "packing": False,
        "dataset_num_proc": None,
        "run_name": cfg.run_name,
        "project": cfg.project,
    }

    if max_steps is not None:
        kwargs["max_steps"] = max_steps
        kwargs["save_steps"] = min(cfg.eval.save_steps, max_steps)

    if do_eval:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = (
            min(cfg.eval.eval_steps, max_steps) if max_steps is not None else cfg.eval.eval_steps
        )
    else:
        kwargs["eval_strategy"] = "no"

    return SFTConfig(**kwargs)


def _model_init_kwargs(cfg: ExperimentConfig) -> dict:
    kwargs: dict = {}
    if cfg.training.fp16:
        kwargs["torch_dtype"] = torch.float16
    elif cfg.training.bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


def _lora_config(cfg: ExperimentConfig) -> LoraConfig | None:
    if not cfg.lora.enabled:
        return None
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )


def main(config_path: str, *, max_steps: int | None = None) -> None:
    cfg_path = Path(config_path).resolve()
    cfg = ExperimentConfig.from_yaml(str(cfg_path))

    train_rows = read_jsonl(_abs_data_path(cfg.data.train_path))
    train_data = _rows_to_conv_sft(train_rows, response_key="response")
    train_dataset = Dataset.from_list(train_data)

    smoke = max_steps is not None
    # Full eval on ~2.8k examples is hundreds of forward passes; skip for --max_steps smoke runs.
    eval_for_trainer: Dataset | None = None
    if not smoke:
        eval_rows = read_jsonl(_abs_data_path(cfg.data.eval_path))
        eval_data = _rows_to_conv_sft(eval_rows, response_key="ground_truth")
        eval_for_trainer = Dataset.from_list(eval_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.primary)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_args = _build_sft_config(cfg, max_steps=max_steps, do_eval=not smoke)
    sft_args.model_init_kwargs = _model_init_kwargs(cfg)

    peft_config = _lora_config(cfg)

    trainer = SFTTrainer(
        model=cfg.model.primary,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_for_trainer,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    # Ensure final weights on disk even if training stopped before save_steps.
    trainer.save_model(sft_args.output_dir)
    tokenizer.save_pretrained(sft_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="If set, train for exactly this many steps (overrides epochs) for smoke tests.",
    )
    args = parser.parse_args()
    main(args.config, max_steps=args.max_steps)
