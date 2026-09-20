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


def test_cached_decode_logits_match_full_forward():
    """Prefill + one cached decode must match a full causal forward on the
    concatenated sequence. That is the KV-cache correctness check."""
    from model import TinyGPT

    torch.manual_seed(0)
    model = TinyGPT(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=8, dropout=0.0)
    model.eval()

    prompt = torch.randint(0, 32, (2, 5))
    new_tok = torch.randint(0, 32, (2, 1))
    full = torch.cat([prompt, new_tok], dim=1)

    with torch.no_grad():
        logits_full, _ = model(full)
        logits_prefill, caches = model._forward_with_cache(prompt, start_pos=0, kv_caches=None)
        logits_decode, caches = model._forward_with_cache(
            new_tok, start_pos=prompt.size(1), kv_caches=caches
        )

    assert logits_prefill.shape == (2, 5, 32)
    assert logits_decode.shape == (2, 1, 32)
    assert caches[0][0].shape == (2, 2, 6, 4)  # B, H, L, d_h
    assert torch.allclose(logits_full[:, :5, :], logits_prefill, atol=1e-5)
    assert torch.allclose(logits_full[:, -1:, :], logits_decode, atol=1e-5)


def test_generate_cached_matches_windowed_greedy():
    from model import TinyGPT

    torch.manual_seed(1)
    model = TinyGPT(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=8, dropout=0.0)
    model.eval()
    prompt = torch.randint(0, 32, (1, 4))

    with torch.no_grad():
        cached = model.generate(prompt.clone(), max_new_tokens=5, temperature=0.0)
        windowed = model._generate_windowed(prompt.clone(), 5, temperature=0.0, top_k=None, eos_id=None)

    assert cached.shape == windowed.shape
    assert torch.equal(cached, windowed)
