"""
Greedy generation + task metrics on a fixed prompt list (shared by eval scripts).
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.common.generation import build_generation_config, decode_assistant_completion
from src.evals.exact_match import (
    avg_response_length,
    entity_micro_prf1,
    exact_field_presence_rate,
    format_validity_rate,
    json_structural_match_rate,
)


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> list[str]:
    """Greedy (or sampled) chat completions — assistant tokens only."""

    outs: list[str] = []
    with torch.no_grad():
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            inputs = inputs.to(device)
            input_ids = inputs["input_ids"]
            gen_cfg = build_generation_config(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            sequences = model.generate(**inputs, generation_config=gen_cfg)
            outs.append(decode_assistant_completion(tokenizer, input_ids, sequences))
    return outs


def score_generations(
    outputs: list[str],
    ground_truths: list[str] | None,
    *,
    required_fields: list[str] | None,
) -> dict[str, float]:
    """Compute format + task metrics for one batch of generations."""

    metrics: dict[str, float] = {
        "format_validity_rate": float(format_validity_rate(outputs)),
        "avg_response_length": float(avg_response_length(outputs)),
    }
    if required_fields:
        metrics["exact_field_presence_rate"] = float(
            exact_field_presence_rate(outputs, required_fields)
        )
    if ground_truths is not None and len(ground_truths) == len(outputs):
        metrics["task_success_rate"] = float(json_structural_match_rate(outputs, ground_truths))
        p, r, f1 = entity_micro_prf1(outputs, ground_truths)
        metrics["entity_precision"] = p
        metrics["entity_recall"] = r
        metrics["entity_f1"] = f1
    return metrics
