"""Sum of response-token log-probabilities.

DPO compares this quantity between a chosen line and a rejected line.
GRPO weights the same quantity by a group advantage. It lives here so the
GRPO trainer can import it without importing the DPO training loop.
"""

import torch
import torch.nn.functional as F

from sft_bpe_data import IGNORE_INDEX


def sequence_logprobs(model, input_ids, labels, attention_mask):
    """Sum of log p(token | context) over response positions only (labels
    != IGNORE_INDEX)."""
    logits, _ = model(input_ids, key_padding_mask=attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != IGNORE_INDEX
    safe_labels = labels.clone()
    safe_labels[~mask] = 0  # placeholder id — zeroed out below, value never used
    token_logprobs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=1)
