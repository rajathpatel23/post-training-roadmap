"""Tiny decoder-only transformer, written from scratch for readability.

Architecture: token embedding + learned positional embedding, N pre-norm
Transformer blocks (causal multi-head self-attention + GELU MLP), final
layer norm, tied-to-nothing output head. ~0.3M params at default config.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (B, H, T, D)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # scaled dot-product
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(causal, float("-inf"))             # no peeking at the future
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.mlp = MLP(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))   # pre-norm residual
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layer=2, n_head=4,
                 d_ff=256, max_seq=128, dropout=0.0):
        super().__init__()
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        B, T = ids.shape
        assert T <= self.max_seq, f"sequence length {T} exceeds max_seq {self.max_seq}"
        x = self.tok_emb(ids) + self.pos_emb(
            torch.arange(T, device=ids.device))[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))  # (B, T, vocab)


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0,
             top_k=0, generator=None, eos_id=2):
    """Sample a completion token-by-token (no KV cache; fine at this scale).

    Returns (completion_ids, old_logprobs): per-token log-probabilities under
    the *sampling* policy. GRPO needs these as the denominator of the
    importance ratio pi_new / pi_old.
    """
    model.eval()
    ids = list(prompt_ids)
    logprobs = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids[-model.max_seq:]]))[0, -1]
        if temperature == 0.0:                       # greedy (used by eval)
            nxt, lp = int(logits.argmax()), 0.0
        else:
            logits = logits / temperature
            if top_k > 0:                            # keep only the top-k logits
                thresh = torch.topk(logits, top_k).values[-1]
                logits[logits < thresh] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
            lp = float(torch.log(probs[nxt]))
        ids.append(nxt)
        logprobs.append(lp)
        if nxt == eos_id:
            break
    return ids[len(prompt_ids):], torch.tensor(logprobs)
