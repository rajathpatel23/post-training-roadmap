"""
Shared generation callback for TinyGPT training scripts — was duplicated
near-identically in train.py and train_sft.py (same encode/generate/decode
body, only the max_new_tokens default differed).
"""

import torch


@torch.no_grad()
def sample_prompts(model, tokenizer, prompts, device, max_new_tokens=120):
    """Run each fixed prompt through the model, decode the completion.
    Used as a qualitative callback every eval step so improvement (or, for
    SFT, the base-vs-trained behavioral change) is visible alongside the
    loss curve, not just implied by a number."""
    model.eval()
    rows = []
    for p in prompts:
        idx = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        out_ids = model.generate(
            idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40, eos_id=tokenizer.eos_id
        )[0].tolist()
        rows.append({"prompt": p, "output": tokenizer.decode(out_ids)})
    model.train()
    return rows


@torch.no_grad()
def sample_prompts_multi(model, tokenizer, prompts, device, k=3, max_new_tokens=120):
    """Same as sample_prompts, but k stochastic draws per prompt instead of
    one. A single draw at temperature=0.8 can't distinguish "the policy
    degraded" from "this was one unlucky low-probability sample" — this
    exists specifically to tell those two apart (see the DPO gold-dataset
    run's collapse diagnosis in reports/scratch_nanogpt.md)."""
    model.eval()
    rows = []
    for p in prompts:
        idx = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        for draw in range(k):
            out_ids = model.generate(
                idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40, eos_id=tokenizer.eos_id
            )[0].tolist()
            rows.append({"prompt": p, "draw": draw, "output": tokenizer.decode(out_ids)})
    model.train()
    return rows
