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
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.scaled_attention = ScaledAttention(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        B, L_q, _ = query.shape  # [B, L_q, D]
        _, L_k, _ = key.shape    # [B, L_k, D]

        query = self.q_linear(query)  # [B, L_q, D]
        key = self.k_linear(key)      # [B, L_k, D]
        value = self.v_linear(value)  # [B, L_k, D]

        # split D into (H, d_k), then put H next to batch so matmul is over (L, d_k)
        query = query.view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_q, d_k]
        key = key.view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)      # [B, H, L_k, d_k]
        value = value.view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_k, d_k]

        output = self.scaled_attention(query, key, value, mask)  # [B, H, L_q, d_k]
        # transpose back, then concat heads along the feature dim
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)  # [B, L_q, D]
        return self.out_linear(output)  # [B, L_q, D]


if __name__ == "__main__":
    B, L, D, H = 2, 10, 512, 8
    query = torch.randn(B, L, D)
    key = torch.randn(B, L, D)
    value = torch.randn(B, L, D)
    mask = torch.ones(B, 1, L, L)
    out = MultiHeadAttention(D, H)(query, key, value, mask)
    assert out.shape == (B, L, D)
    print(out.shape)
