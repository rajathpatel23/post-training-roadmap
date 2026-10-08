import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledAttention(nn.Module):
    """Vaswani Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    No W_Q / W_K / W_V here — those live on MultiHeadAttention.
    Last two dims of Q,K,V must be (L, d_k).
    """

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        # query: [B, H, L_q, d_k]
        # key:   [B, H, L_k, d_k]
        # value: [B, H, L_k, d_k]
        # mask:  broadcastable to [B, H, L_q, L_k]  (1 = keep, 0 = block)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L_q, L_k]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # [B, H, L_q, L_k]
        return torch.matmul(weights, value)  # [B, H, L_q, d_k]


class MultiHeadAttention(nn.Module):
    """Vaswani MultiHead(Q, K, V) = Concat(head_1..head_h) W_O
    where head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i).
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # D → D is the same as H separate maps D → d_k, then concat
        # these are attention weights for each head
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.scaled_attention = ScaledAttention(self.d_k)

    def _project(self, x: torch.Tensor):
        # x: [B, L, D] -> q,k,v each [B, H, L, d_k]
        # same weights for prefill (L tokens) and later decode (1 token)
        B, L, _ = x.shape
        q = self.q_linear(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        return q, k, v

    def _merge(self, heads: torch.Tensor):
        # [B, H, L, d_k] -> [B, L, D]
        B, H, L, d_k = heads.shape
        return heads.transpose(1, 2).contiguous().view(B, L, H * d_k)

    def prefill(self, x: torch.Tensor):
        # Prompt self-attention. Q is used this pass, then dropped.
        # K,V are the cache Exercise 3 will concat onto.
        B, L, _ = x.shape
        q, k, v = self._project(x)
        causal = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        out = self.scaled_attention(q, k, v, causal)  # [B, H, L, d_k]
        out = self.out_linear(self._merge(out))       # [B, L, D]
        return out, k, v                             # k,v: [B, H, L, d_k]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        B, L_q, _ = query.shape  # [B, L_q, D]
        # training path: no cache. Q/K/V may differ (cross-attn).
        q = self.q_linear(query).view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(B, key.size(1), self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(B, value.size(1), self.n_heads, self.d_k).transpose(1, 2)

        output = self.scaled_attention(q, k, v, mask)  # [B, H, L_q, d_k]
        return self.out_linear(self._merge(output))     # [B, L_q, D]


if __name__ == "__main__":
    B, L, D, H = 2, 10, 512, 8
    query = torch.randn(B, L, D)
    key = torch.randn(B, L, D)
    value = torch.randn(B, L, D)
    mask =torch.tril(torch.ones(L, L)).view(1, 1, L, L)
    out = MultiHeadAttention(D, H)(query, key, value, mask)
    assert out.shape == (B, L, D)
    print(out.shape)
    torch.manual_seed(0)
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, L, D)
    causal = torch.tril(torch.ones(L, L)).view(1, 1, L, L)  # 1 = keep, 0 = future
    y1 = mha(x, x, x, causal)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(B, L - 5, D)  # change only positions 5..9
    y2 = mha(x2, x2, x2, causal)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)  # past unaffected by future
    assert not torch.allclose(y1[:, 5:], y2[:, 5:])          # future did change
    print("causal ok")

    out_pf, k_cache, v_cache = mha.prefill(x)
    assert out_pf.shape == (B, L, D)
    assert k_cache.shape == (B, H, L, D // H)
    assert v_cache.shape == (B, H, L, D // H)
    assert torch.allclose(y1, out_pf, atol=1e-6)
    print("prefill matches causal forward")