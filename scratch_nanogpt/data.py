"""
Character-level tokenizer + train/val split for tiny-shakespeare.

Deliberately not BPE — char-level keeps the vocabulary tiny (~65 symbols) and
the tokenizer trivial, so all the training-loop complexity is in the model,
not the data pipeline. Swap in a real tokenizer later once the training loop
itself is understood.

Reserves one special id (0) for [EOS] — not a real character, never appears
in the raw pretraining corpus (tiny-shakespeare has no document boundaries to
place it at), so its embedding row starts and stays untrained through
pretraining. SFT is the stage that actually teaches the model what it means,
by appending it after every well-formed response.
"""

import os

import torch

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")

EOS_TOKEN = "[EOS]"


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.eos_id = 0
        self.stoi = {EOS_TOKEN: self.eos_id}
        offset = 1
        for i, ch in enumerate(chars):
            self.stoi[ch] = i + offset
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "?") if i != self.eos_id else "" for i in ids)


def load_data(val_fraction: float = 0.1):
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(len(data) * (1 - val_fraction))
    train_data = data[:n]
    val_data = data[n:]
    return tokenizer, train_data, val_data


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
