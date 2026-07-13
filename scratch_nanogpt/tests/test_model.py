"""
Tests for model.py's Attention — specifically that causal masking and
padding masking combine correctly (needed for SFT's variable-length
batches). This is exactly the kind of combinatorial masking logic that can
be subtly wrong in a way that still "trains fine" but silently leaks
information through padding — hard to catch from a loss curve, easy to
catch with a direct test.
"""

import torch

from model import Attention


def test_padding_mask_blocks_attention_to_padded_keys():
    """A later, real (non-padded) position must be unaffected by changes at
    an earlier position that's marked as padding — even though causal
    masking alone would otherwise allow attending to it."""
    torch.manual_seed(0)
    attn = Attention(n_embd=8, n_head=2, block_size=8, dropout=0.0, causal=True)
    attn.eval()

    x = torch.randn(1, 4, 8)
    # Positions 0, 1 are padding; 2, 3 are real content.
    key_padding_mask = torch.tensor([[False, False, True, True]])

    with torch.no_grad():
        out_before = attn(x, key_padding_mask=key_padding_mask)

    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 100.0
    x_perturbed[:, 1, :] += 100.0

    with torch.no_grad():
        out_after = attn(x_perturbed, key_padding_mask=key_padding_mask)

    # Positions 2 and 3 are causally allowed to see 0/1, but the padding
    # mask should block that — their output must not change.
    assert torch.allclose(out_before[:, 2, :], out_after[:, 2, :], atol=1e-5)
    assert torch.allclose(out_before[:, 3, :], out_after[:, 3, :], atol=1e-5)


def test_without_padding_mask_causal_attention_does_leak_earlier_positions():
    """Sanity check that the test above is actually meaningful: with no
    padding mask, a later position's output DOES change when an earlier
    position changes (that's normal causal attention, not a bug) — confirms
    the previous test isn't vacuously passing regardless of the mask."""
    torch.manual_seed(0)
    attn = Attention(n_embd=8, n_head=2, block_size=8, dropout=0.0, causal=True)
    attn.eval()

    x = torch.randn(1, 4, 8)

    with torch.no_grad():
        out_before = attn(x, key_padding_mask=None)

    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 100.0

    with torch.no_grad():
        out_after = attn(x_perturbed, key_padding_mask=None)

    assert not torch.allclose(out_before[:, 3, :], out_after[:, 3, :], atol=1e-5)


def test_causal_mask_alone_blocks_future_positions():
    """Position i must never be affected by positions after it, regardless
    of padding — the basic causal property, unrelated to padding."""
    torch.manual_seed(0)
    attn = Attention(n_embd=8, n_head=2, block_size=8, dropout=0.0, causal=True)
    attn.eval()

    x = torch.randn(1, 4, 8)

    with torch.no_grad():
        out_before = attn(x, key_padding_mask=None)

    x_perturbed = x.clone()
    x_perturbed[:, 3, :] += 100.0  # perturb the last (future-most) position

    with torch.no_grad():
        out_after = attn(x_perturbed, key_padding_mask=None)

    # Positions 0, 1, 2 come before position 3 — causally forbidden from seeing it.
    assert torch.allclose(out_before[:, :3, :], out_after[:, :3, :], atol=1e-5)
