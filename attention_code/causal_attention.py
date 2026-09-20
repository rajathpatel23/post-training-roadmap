import torch
import torch.nn as nn
from scaled_attention import ScaledAttention

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CausalAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.scaled_attention = ScaledAttention(d_model, n_heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ):
        B, L, D = query.shape
        _, _, D = key.shape
        _, _, D = value.shape

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        # 1 = allowed, 0 = blocked. [L, L] broadcasts over [B, H, L, L] scores.
        causal_mask = torch.tril(torch.ones(L, L, device=query.device))
        if mask is not None:
            mask = mask * causal_mask
        else:
            mask = causal_mask

        return self.scaled_attention(query, key, value, mask, return_weights=return_weights)

if __name__ == "__main__":
    query = torch.randn(2, 10, 512)
    key = torch.randn(2, 10, 512)
    value = torch.randn(2, 10, 512)
    causal_attention = CausalAttention(512, 8)
    output, attn_weights = causal_attention(query, key, value, return_weights=True)
    assert output.shape == (2, 10, 512)

    # attn_weights: [B, H, L, L]. Position i must not attend to j > i.
    future = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    print(future)
    assert torch.allclose(attn_weights[:, :, future], torch.zeros(()), atol=1e-6)
    print(attn_weights)
    print(output.shape)
    print("upper triangle after softmax:", attn_weights[0, 0, future].abs().max().item())