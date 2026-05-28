"""
Load causal LMs for inference — full checkpoints or PEFT LoRA adapters.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def is_peft_adapter_path(model_path: str | Path) -> bool:
    """True when `model_path` contains a PEFT adapter (not a merged full model)."""

    return (Path(model_path) / "adapter_config.json").is_file()


def _resolve_torch_dtype(*, fp16: bool, bf16: bool) -> torch.dtype | None:
    if fp16:
        return torch.float16
    if bf16:
        return torch.bfloat16
    return None


def load_tokenizer(
    model_path: str,
    *,
    base_model: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load tokenizer from checkpoint dir, falling back to base model if needed."""

    path = Path(model_path)
    if (path / "tokenizer_config.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif base_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm_for_inference(
    model_path: str,
    *,
    base_model: str | None = None,
    device: str,
    fp16: bool = False,
    bf16: bool = False,
) -> PreTrainedModel:
    """
    Load a model for greedy/sampled generation.

    - Merged or base HF checkpoints: `AutoModelForCausalLM.from_pretrained(model_path)`
    - LoRA adapter dirs: load `base_model` + `PeftModel.from_pretrained(...)`
    """

    dtype = _resolve_torch_dtype(fp16=fp16, bf16=bf16)
    model_kwargs: dict = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if is_peft_adapter_path(model_path):
        if base_model is None:
            raise ValueError(
                f"{model_path} is a PEFT adapter; pass base_model=... (e.g. cfg.model.primary)"
            )
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.eval()
    model.to(device)
    return model


def merge_lora_and_save(
    adapter_path: str,
    *,
    base_model: str,
    output_dir: str,
    fp16: bool = False,
    bf16: bool = False,
) -> str:
    """
    Bake LoRA weights into the base model and write a standalone HF checkpoint.

    Returns the absolute path to `output_dir`.
    """

    if not is_peft_adapter_path(adapter_path):
        raise ValueError(f"Expected PEFT adapter at {adapter_path}")

    dtype = _resolve_torch_dtype(fp16=fp16, bf16=bf16)
    model_kwargs: dict = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tokenizer = load_tokenizer(adapter_path, base_model=base_model)
    tokenizer.save_pretrained(out)
    return str(out.resolve())


def release_model(model: PreTrainedModel) -> None:
    """Drop model references and clear MPS/CUDA cache when available."""

    del model
    try:
        torch.mps.empty_cache()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
