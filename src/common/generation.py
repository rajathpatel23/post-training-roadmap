"""
Helpers for causal LM inference — decode only the assistant completion, not the prompt.
"""

from __future__ import annotations

import torch
from transformers import GenerationConfig, PreTrainedTokenizerBase


def build_generation_config(
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> GenerationConfig:
    """
    Build configs that match how we decode — greedy omits sampling-only fields.

    If greedy decoding still merges temperature/top_p/top_k from the model,
    Hugging Face warns that those flags are ignored; omitting them on the
    explicit config avoids noisy stderr when ``do_sample`` is False.
    """

    kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    return GenerationConfig(**kwargs)


def decode_assistant_completion(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    sequences: torch.Tensor,
) -> str:
    """
    Decode tokens generated after the chat prompt (batch size 1).

    `input_ids` is the tensor passed into `model.generate` (prompt only).
    `sequences` is the full output including the prompt prefix.
    """

    prompt_len = int(input_ids.shape[-1])
    new_tokens = sequences[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def resolve_lm_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
