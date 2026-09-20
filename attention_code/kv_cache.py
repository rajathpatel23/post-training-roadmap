import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None):
    """Scores + softmax + weighted V. No projections.

    query: [B, H, Lq, d_h]
    key:   [B, H, Lk, d_h]
    value: [B, H, Lk, d_h]
    mask:  1 = allowed, 0 = blocked. Broadcasts onto [B, H, Lq, Lk].
    """
    d_h = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_h)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)


def decode_step(q_new, k_new, v_new, k_cache, v_cache):
    """Concat new K/V onto the cache, attend with only q_new.

    All tensors are already projected and split: [B, H, L, d_h].
    """
    k_cache = torch.cat([k_cache, k_new], dim=2)
    v_cache = torch.cat([v_cache, v_new], dim=2)
    out = attention(q_new, k_cache, v_cache)
    return out, k_cache, v_cache


def split_heads(x, n_heads):
    # [B, L, D] -> [B, H, L, d_h]
    B, L, D = x.shape
    return x.view(B, L, n_heads, D // n_heads).transpose(1, 2)


def merge_heads(x):
    # [B, H, L, d_h] -> [B, L, D]
    B, H, L, d_h = x.shape
    return x.transpose(1, 2).contiguous().view(B, L, H * d_h)


class CachedAttention(nn.Module):
    """One attention layer. Owns Q/K/V linears. Does not store the cache.

    prefill(x): x is the prompt [B, L, D]. Full causal attention, return K/V.
    decode(x_new, k, v): x_new is one token [B, 1, D]. Reuse cached K/V.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

    def _project(self, x):
        # Same weights for prefill (L tokens) and decode (1 token).
        q = split_heads(self.q_linear(x), self.n_heads)
        k = split_heads(self.k_linear(x), self.n_heads)
        v = split_heads(self.v_linear(x), self.n_heads)
        return q, k, v

    def prefill(self, x):
        B, L, _ = x.shape
        q, k, v = self._project(x)
        causal_mask = torch.tril(torch.ones(L, L, device=x.device))
        out = attention(q, k, v, mask=causal_mask)
        return merge_heads(out), k, v

    def decode(self, x_new, k_cache, v_cache):
        q_new, k_new, v_new = self._project(x_new)
        out, k_cache, v_cache = decode_step(q_new, k_new, v_new, k_cache, v_cache)
        return merge_heads(out), k_cache, v_cache


if __name__ == "__main__":
    B, L, D, H = 2, 10, 512, 8
    layer = CachedAttention(D, H)

    x_prompt = torch.randn(B, L, D)
    x_new = torch.randn(B, 1, D)
    x_full = torch.cat([x_prompt, x_new], dim=1)

    # Prefill of the full sequence: last position is the "new" token.
    out_full, k_full, v_full = layer.prefill(x_full)

    # Same last token, but via cache: prefill prompt, then decode one step.
    out_prompt, k_cache, v_cache = layer.prefill(x_prompt)
    out_decode, k_cache, v_cache = layer.decode(x_new, k_cache, v_cache)

    assert out_prompt.shape == (B, L, D)
    assert out_decode.shape == (B, 1, D)
    assert k_cache.shape == (B, H, L + 1, D // H)

    # Causal last-position == decode. That is the KV-cache correctness check.
    assert torch.allclose(out_full[:, -1:, :], out_decode, atol=1e-5)
    assert torch.allclose(k_full, k_cache, atol=1e-6)
    assert torch.allclose(v_full, v_cache, atol=1e-6)
    print("prefill + decode match last-token prefill", out_decode.shape)
