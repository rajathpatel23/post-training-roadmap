import torch
import torch.nn as nn
import torch.nn.functional as F
from scaled_attention import ScaledAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.scaled_attention = ScaledAttention(d_model, n_heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        B, L, D = query.shape
        _, _, D = key.shape
        _, _, D = value.shape

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        print(mask.shape)
        output = self.scaled_attention(query, key, value, mask)
        return output

if __name__ == "__main__":
    query = torch.randn(2, 10, 512)
    key = torch.randn(2, 10, 512)
    value = torch.randn(2, 10, 512)
    mask = torch.randn(2, 8, 10, 10)
    multi_head_attention = MultiHeadAttention(512, 8)
    output = multi_head_attention(query, key, value, mask=mask)
    assert output.shape == (2, 10, 512)
