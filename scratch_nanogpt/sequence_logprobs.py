"""Response-token log-probabilities (sum or length-normalized mean).

DPO compares *sums* between chosen and rejected (sequence-level margin).
GRPO can weight the same tokens by advantage using a *mean* so longer
completions don't dominate the gradient. Shared helper so GRPO doesn't
import the DPO training loop.
"""

import torch
import torch.nn.functional as F

from sft_bpe_data import IGNORE_INDEX


def sequence_logprobs(model, input_ids, labels, attention_mask, reduction: str = "sum"):
    """log p over response positions only (labels != IGNORE_INDEX).

    reduction: "sum" (DPO default) or "mean" (per-response-token, for GRPO).
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
    logits, _ = model(input_ids, key_padding_mask=attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != IGNORE_INDEX
    safe_labels = labels.clone()
    safe_labels[~mask] = 0  # placeholder id — zeroed out below, value never used
    token_logprobs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    summed = token_logprobs.sum(dim=1)
    if reduction == "sum":
        return summed
    lengths = mask.sum(dim=1).clamp(min=1)
    return summed / lengths
