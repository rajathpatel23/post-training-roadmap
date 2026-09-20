import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, L, D = query.shape
        _, _, D = key.shape
        _, _, D = value.shape

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value) 
        import pdb; pdb.set_trace()
        query = query.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        key = key.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        value = value.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        import pdb; pdb.set_trace()
        output = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        output = F.softmax(output, dim=-1)
        output = torch.matmul(output, value)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return output


if __name__ == "__main__":
    query = torch.randn(1, 10, 512)
    key = torch.randn(1, 10, 512)
    value = torch.randn(1, 10, 512)
    attention = Attention(512, 8)
    output = attention(query, key, value)
    print(output.shape)