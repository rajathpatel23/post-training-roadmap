"""
Helpers for causal LM inference — decode only the assistant completion, not the prompt.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


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
