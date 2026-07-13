"""
Minimal transformers, written from scratch (no HF, no TRL).

Purpose: understand every piece by building it, not fine-tune a pretrained model.
Char-level on tiny-shakespeare — small enough to train on CPU/MPS in minutes.

Two models share the same Block/MLP building blocks:
- TinyGPT:  causal self-attention, next-token loss (decoder-only, GPT-2/Qwen/Llama shape)
- TinyBERT: bidirectional self-attention, MLM + NSP loss (encoder-only, BERT shape)

The only structural difference between them is the attention mask:
- GPT:  position i can only see positions <= i (causal_mask, fixed at init)
- BERT: every position sees every non-padding position (key_padding_mask, varies per batch)
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Multi-head self-attention. `causal=True` builds a fixed lower-triangular
    mask (GPT). `causal=False` expects an optional per-batch key_padding_mask
    instead (BERT) — every real token attends to every other real token."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, causal: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if causal:
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """key_padding_mask: (B, T) bool, True = real token, False = padding.
        For GPT (causal=True), used for batched variable-length sequences
        (e.g. SFT prompt+response pairs) — combined with the causal mask, not
        a replacement for it. For BERT (causal=False), it's the only mask."""
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if self.causal:
            blocked = self.causal_mask[:, :, :T, :T] == 0
            if key_padding_mask is not None:
                blocked = blocked | ~key_padding_mask.view(B, 1, 1, T)
            att = att.masked_fill(blocked, float("-inf"))
        elif key_padding_mask is not None:
            # Block attention *to* padding positions (key dim), for every query and every head.
            mask = key_padding_mask.view(B, 1, 1, T)
            att = att.masked_fill(~mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-norm transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, causal: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head, block_size, dropout, causal=causal)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, causal=True) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None):
        """key_padding_mask: (B, T) bool, True = real token, False = padding —
        for batched variable-length sequences (e.g. SFT prompt+response pairs).
        targets: -100 at any position that shouldn't contribute to the loss
        (SFT: prompt tokens and padding) — F.cross_entropy ignores -100 by
        default, same convention TinyBERT already uses for MLM."""
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ):
        """eos_id: stop once every sequence in the batch has sampled it (real
        early stopping, not just a soft bias) — max_new_tokens is still the
        hard cap if eos_id is None or never sampled."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and (idx_next == eos_id).all():
                break
        return idx


class TinyBERT(nn.Module):
    """Encoder-only: bidirectional attention, MLM + NSP joint pretraining.

    Input layout per example: [CLS] segA [SEP] segB [SEP] (padded to block_size).
    - MLM head: token-level, predicts original token at masked positions only.
    - NSP head: pools the [CLS] position's final hidden state -> binary IsNext/NotNext.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.seg_emb = nn.Embedding(2, n_embd)  # segment A=0 / B=1
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, causal=False) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

        self.mlm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.pooler = nn.Sequential(nn.Linear(n_embd, n_embd), nn.Tanh())
        self.nsp_head = nn.Linear(n_embd, 2)

        self.apply(_init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mlm_labels: torch.Tensor | None = None,
        nsp_labels: torch.Tensor | None = None,
    ):
        """
        input_ids, segment_ids, attention_mask: (B, T)
        attention_mask: 1 = real token, 0 = padding
        mlm_labels: (B, T), -100 at non-masked positions (ignored by cross_entropy)
        nsp_labels: (B,), 0 = IsNext, 1 = NotNext
        """
        B, T = input_ids.shape
        assert T <= self.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos) + self.seg_emb(segment_ids)
        x = self.drop(x)

        key_padding_mask = attention_mask.bool()
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)

        mlm_logits = self.mlm_head(x)  # (B, T, vocab_size)

        cls_hidden = x[:, 0, :]  # [CLS] is always position 0
        nsp_logits = self.nsp_head(self.pooler(cls_hidden))  # (B, 2)

        mlm_loss = None
        if mlm_labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1), ignore_index=-100
            )

        nsp_loss = None
        if nsp_labels is not None:
            nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)

        return {
            "mlm_logits": mlm_logits,
            "nsp_logits": nsp_logits,
            "mlm_loss": mlm_loss,
            "nsp_loss": nsp_loss,
        }
