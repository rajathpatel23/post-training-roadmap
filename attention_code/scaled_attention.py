import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(ScaledAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        return_weights: bool = False,
    ):
        B, L, D = query.shape
        _, _, D = key.shape
        _, _, D = value.shape

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        key = key.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        value = value.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        if return_weights:
            return output, attn_weights
        return output

if __name__ == "__main__":
    query = torch.randn(1, 10, 512)
    key = torch.randn(1, 10, 512)
    value = torch.randn(1, 10, 512)
    mask = torch.randn(1, 10, 10)
    scaled_attention = ScaledAttention(512, 8)
    output = scaled_attention(query, key, value, mask=mask)
    print(query[0][0].mean())
    print(key[0][0].mean())
    print(value[0][0].mean())
    print(mask[0][0].mean())
    print(output[0][0].mean())