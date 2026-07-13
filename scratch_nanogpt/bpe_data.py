"""
GPT-2 BPE tokenizer + data loading — drop-in alternative to data.py's
char-level tokenizer. Same load_data() return shape (tokenizer, train_data,
val_data) so train.py switches between char/BPE via a config flag, not a
code change. get_batch() is untouched — it just samples fixed-length windows
from a 1D LongTensor and doesn't care what the ids represent.

Reuses the actual pretrained GPT-2 vocab (50,257 tokens, byte-level BPE) —
not retraining BPE on tiny-shakespeare. GPT-3's original models (davinci
etc.) used this same vocab, so this is also "GPT-3's BPE."

data_path is a parameter (not a hardcoded constant) so the same tokenizer
can encode a *different* corpus (e.g. movie_dialogue.txt for continued
pretraining) without re-downloading GPT-2's tokenizer files each time —
see encode_corpus(), used for the Shakespeare-vs-movie forgetting check.
"""

import os

import torch
from transformers import GPT2TokenizerFast

SHAKESPEARE_PATH = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
MOVIE_PATH = os.path.join(os.path.dirname(__file__), "data", "movie_dialogue.txt")


class BPETokenizer:
    """Wraps HF's GPT2TokenizerFast so it exposes the same .encode()/.decode()/
    .vocab_size surface as data.py's CharTokenizer — train.py's sample_prompts()
    callback works unchanged regardless of which tokenizer is active."""

    def __init__(self):
        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.vocab_size = self._tok.vocab_size
        self.eos_id = self._tok.eos_token_id  # GPT-2 already has one: <|endoftext|>, id 50256

    def encode(self, s: str) -> list[int]:
        return self._tok.encode(s)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def encode_corpus(tokenizer: BPETokenizer, data_path: str, val_fraction: float = 0.1):
    """Encode a corpus with an already-loaded tokenizer — reused for scoring a
    second corpus (e.g. the original Shakespeare val set) without reloading
    GPT-2's tokenizer files again."""
    with open(data_path, encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    print(f"BPE: {len(text):,} chars -> {len(ids):,} tokens ({len(text) / len(ids):.2f} chars/token) [{data_path}]")
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]


def load_data(data_path: str = SHAKESPEARE_PATH, val_fraction: float = 0.1):
    tokenizer = BPETokenizer()
    train_data, val_data = encode_corpus(tokenizer, data_path, val_fraction)
    return tokenizer, train_data, val_data
