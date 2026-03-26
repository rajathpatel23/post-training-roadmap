from __future__ import annotations

"""
Minimal experiment config system.

Goals:
- Load `configs/<project>.yaml` which can specify `defaults: [base]`.
- Deep-merge base + project overrides.
- Provide dataclass access for the keys we care about (and tolerate missing ones).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import yaml


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(obj)}")
    return obj


@dataclass(frozen=True)
class ModelConfig:
    primary: str
    backup: Optional[str] = None
    use_backup: bool = False


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    bias: str
    task_type: str


@dataclass(frozen=True)
class TrainingConfig:
    max_length: int
    grad_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler: str
    fp16: bool
    bf16: bool
    dataloader_num_workers: int
    seed: int

    # Project-specific training params (defined in project YAML)
    learning_rate: float
    per_device_train_batch_size: int
    num_train_epochs: int
    output_dir: str


@dataclass(frozen=True)
class EvalConfig:
    eval_steps: int
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    generation_max_new_tokens: int
    generation_temperature: float
    generation_do_sample: bool
    num_sample_generations: int

    # Optional extra knobs (not present in current base.yaml)
    required_fields: Optional[List[str]] = None


@dataclass(frozen=True)
class LoggingConfig:
    report_to: str
    logging_steps: int


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    eval_path: str


@dataclass(frozen=True)
class ExperimentConfig:
    project: str
    run_name: str
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    eval: EvalConfig
    logging: LoggingConfig
    data: DataConfig
    # Keep a copy of anything else (e.g., dpo/grpo subtrees) for scripts that need it.
    extra: Dict[str, Any]

    @staticmethod
    def from_yaml(
        config_path: str,
        *,
        configs_dir: str = "configs",
    ) -> "ExperimentConfig":
        config_path = os.path.abspath(config_path)
        # Resolve repo root robustly by searching upwards for `configs/base.yaml`.
        cur = os.path.abspath(os.path.dirname(config_path))
        repo_root: Optional[str] = None
        while True:
            candidate = os.path.join(cur, "configs", "base.yaml")
            if os.path.exists(candidate):
                repo_root = cur
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        if repo_root is None:
            raise FileNotFoundError(f"Could not locate repo root containing `configs/base.yaml` from {config_path}")

        configs_dir_abs = os.path.abspath(os.path.join(repo_root, configs_dir))

        override = _load_yaml(config_path)
        defaults = override.pop("defaults", None)
        if defaults is None:
            merged = override
        else:
            if not isinstance(defaults, list):
                raise ValueError(f"`defaults` must be a list in {config_path}")

            merged: Dict[str, Any] = {}
            for entry in defaults:
                if isinstance(entry, str):
                    base_name = entry
                    base_path = os.path.join(configs_dir_abs, f"{base_name}.yaml")
                    merged = _deep_merge_dicts(merged, _load_yaml(base_path))
                elif isinstance(entry, dict):
                    # Support a very small subset: {"base": {...}} or {"base": "..."} patterns.
                    # We keep it strict to avoid “magic” merges.
                    if len(entry) != 1:
                        raise ValueError(f"Unsupported defaults entry in {config_path}: {entry}")
                    (base_name, _payload) = next(iter(entry.items()))
                    base_path = os.path.join(configs_dir_abs, f"{base_name}.yaml")
                    merged = _deep_merge_dicts(merged, _load_yaml(base_path))
                else:
                    raise ValueError(f"Unsupported defaults entry type: {type(entry)}")

            merged = _deep_merge_dicts(merged, override)

        # Pull out the keys we know we need.
        project = str(merged["project"])
        run_name = str(merged["run_name"])

        model = ModelConfig(**merged["model"])
        lora = LoRAConfig(**merged["lora"])

        training = TrainingConfig(
            **{
                # base.yaml keys
                "max_length": merged["training"]["max_length"],
                "grad_accumulation_steps": merged["training"]["grad_accumulation_steps"],
                "warmup_ratio": merged["training"]["warmup_ratio"],
                "weight_decay": merged["training"]["weight_decay"],
                "lr_scheduler": merged["training"]["lr_scheduler"],
                "fp16": merged["training"]["fp16"],
                "bf16": merged["training"]["bf16"],
                "dataloader_num_workers": merged["training"]["dataloader_num_workers"],
                "seed": merged["training"]["seed"],
                # project yaml keys
                "learning_rate": merged["training"]["learning_rate"],
                "per_device_train_batch_size": merged["training"]["per_device_train_batch_size"],
                "num_train_epochs": merged["training"]["num_train_epochs"],
                "output_dir": merged["training"]["output_dir"],
            }
        )

        eval_required_fields = None
        if isinstance(merged.get("eval", {}), dict):
            eval_required_fields = merged["eval"].get("required_fields")

        eval_cfg = EvalConfig(
            eval_steps=merged["eval"]["eval_steps"],
            save_steps=merged["eval"]["save_steps"],
            save_total_limit=merged["eval"]["save_total_limit"],
            load_best_model_at_end=merged["eval"]["load_best_model_at_end"],
            metric_for_best_model=merged["eval"]["metric_for_best_model"],
            generation_max_new_tokens=merged["eval"]["generation_max_new_tokens"],
            generation_temperature=merged["eval"]["generation_temperature"],
            generation_do_sample=merged["eval"]["generation_do_sample"],
            num_sample_generations=merged["eval"]["num_sample_generations"],
            required_fields=eval_required_fields,
        )

        logging_cfg = LoggingConfig(**merged["logging"])
        data_cfg = DataConfig(**merged["data"])

        known_keys = {"project", "run_name", "model", "lora", "training", "eval", "logging", "data"}
        extra = {k: v for k, v in merged.items() if k not in known_keys}

        return ExperimentConfig(
            project=project,
            run_name=run_name,
            model=model,
            lora=lora,
            training=training,
            eval=eval_cfg,
            logging=logging_cfg,
            data=data_cfg,
            extra=extra,
        )

