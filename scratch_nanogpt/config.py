"""
YAML config loading for scratch_nanogpt, with a typed dataclass schema per
training script instead of a plain dict.

Deliberately not reusing src/common/config.py — that one models the HF/LoRA
project schema (per_device_train_batch_size, bf16, etc.) which doesn't apply
to this from-scratch code.

Why dataclasses instead of the plain dict this used to return: the same
`init_from` key meant two different things in train.py (weights to mutate)
vs. an earlier version of train_sft.py (frozen comparison reference) before
that got sorted out — a dict lets any key mean anything, silently, under
time pressure. A typed schema per script makes each script's actual config
surface explicit and catches typos/stale fields at load time instead of
deep inside a training run.

Run a different architecture/training setup by pointing at a different YAML,
not by editing Python:
    python train.py --config configs/train/gpt_small.yaml
    python train_bert.py --config configs/train/bert_small.yaml
    python train_sft.py --config configs/train/sft_bpe.yaml
"""

import os
from dataclasses import dataclass
from typing import Optional

import yaml

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_yaml_merged(train_config_path: str) -> dict:
    """Load a train config YAML, resolving its `model_config: <name>`
    reference to configs/model/<name>.yaml. Returns a plain merged dict —
    raw material for the typed load_*_config() loaders below; training
    scripts should use those, not this, directly."""
    with open(train_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.pop("model_config")
    model_path = os.path.join(CONFIGS_DIR, "model", f"{model_name}.yaml")
    with open(model_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    cfg["model"] = model_cfg
    return cfg


@dataclass(frozen=True)
class ModelConfig:
    type: str  # "gpt" or "bert"
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    tokenizer: str = "char"  # "char" or "bpe" — only meaningful when type == "gpt"


@dataclass(frozen=True)
class GPTTrainConfig:
    """train.py — pretraining or continued pretraining of TinyGPT."""

    model: ModelConfig
    batch_size: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    eval_iters: int
    seed: int
    wandb_project: str
    wandb_run_name: str
    data_path: Optional[str] = None
    init_from: Optional[str] = None
    secondary_eval_path: Optional[str] = None
    output_dir: Optional[str] = None
    continue_history: bool = False


def load_gpt_train_config(path: str) -> GPTTrainConfig:
    raw = _load_yaml_merged(path)
    raw["model"] = ModelConfig(**raw["model"])
    return GPTTrainConfig(**raw)


@dataclass(frozen=True)
class BERTTrainConfig:
    """train_bert.py — pretraining of TinyBERT (MLM + NSP)."""

    model: ModelConfig
    batch_size: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    eval_iters: int
    mlm_probability: float
    seed: int
    wandb_project: str
    wandb_run_name: str
    output_dir: Optional[str] = None


def load_bert_train_config(path: str) -> BERTTrainConfig:
    raw = _load_yaml_merged(path)
    raw["model"] = ModelConfig(**raw["model"])
    return BERTTrainConfig(**raw)


@dataclass(frozen=True)
class SFTTrainConfig:
    """train_sft.py — SFT fine-tuning of a pretrained TinyGPT (bpe only)."""

    model: ModelConfig
    init_from: str
    sft_train_path: str
    sft_eval_path: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    eval_interval: int
    seed: int
    wandb_project: str
    wandb_run_name: str
    reference_checkpoint: Optional[str] = None  # defaults to init_from — see train_sft.py
    continue_history: bool = False
    output_dir: Optional[str] = None


def load_sft_train_config(path: str) -> SFTTrainConfig:
    raw = _load_yaml_merged(path)
    raw["model"] = ModelConfig(**raw["model"])
    return SFTTrainConfig(**raw)


@dataclass(frozen=True)
class DPOTrainConfig:
    """train_dpo.py — DPO preference optimization on top of the SFT checkpoint.

    Needs two checkpoints for two different reasons: sft_checkpoint
    initializes *both* the trainable policy and the frozen reference model
    DPO's loss compares against (standard DPO setup — the reference never
    updates). pretrained_checkpoint is unrelated to training at all — it's
    only used for the three-way base/SFT/DPO qualitative comparison, so the
    full pretrain -> SFT -> DPO trajectory is visible on the same prompts,
    not just SFT -> DPO.
    """

    model: ModelConfig
    sft_checkpoint: str
    pretrained_checkpoint: str
    dpo_train_path: str
    dpo_eval_path: str
    beta: float
    batch_size: int
    learning_rate: float
    num_epochs: int
    eval_interval: int
    seed: int
    wandb_project: str
    wandb_run_name: str
    output_dir: Optional[str] = None


def load_dpo_train_config(path: str) -> DPOTrainConfig:
    raw = _load_yaml_merged(path)
    raw["model"] = ModelConfig(**raw["model"])
    return DPOTrainConfig(**raw)
