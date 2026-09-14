"""Word-level tokenizer with a fixed, tiny, hand-built vocabulary.

No downloads, no training: the vocabulary covers exactly the words our problem
generator emits (data.py), plus single digits 0-9 so numbers are rendered as
space-separated digit tokens, e.g. 42 -> "4 2", -7 -> "minus 7".
"""

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
DIGITS = [str(d) for d in range(10)]
PUNCT = ["?", ".", ":"]
WORDS = [
    "what", "is", "the", "of", "and", "a", "by", "from",
    "plus", "minus", "times", "divided",
    "add", "subtract", "multiply", "divide",
    "sum", "difference", "product",
    "think", "step", "then", "write", "answer", "first", "next",
    "equals", "compute", "result",
]


class Tokenizer:
    def __init__(self):
        self.itos = SPECIAL + DIGITS + PUNCT + WORDS
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> list:
        """Whitespace tokenize; anything out of vocab becomes <unk>."""
        return [self.stoi.get(w, self.unk_id) for w in text.lower().split()]

    def decode(self, ids) -> str:
        toks = [self.itos[i] for i in ids
                if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return " ".join(toks)
