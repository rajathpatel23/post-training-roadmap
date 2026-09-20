"""Walk cached generate on the char-level Shakespeare checkpoint.

From scratch_nanogpt:
    python demo_kv_cache.py
"""

import json
import time

import torch

from model import TinyGPT

TOKENIZER_PATH = "out/tokenizer.json"
CKPT_PATH = "out/tinygpt_shakespeare.pt"
PROMPT = "ROMEO:"
WALK_STEPS = 40
FULL_NEW_TOKENS = 120


def load_tokenizer():
    with open(TOKENIZER_PATH) as f:
        raw = json.load(f)
    stoi = raw["stoi"]
    itos = {int(k): v for k, v in raw["itos"].items()}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    return encode, decode, len(stoi)


def main():
    encode, decode, vocab_size = load_tokenizer()
    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu", weights_only=True))
    model.eval()

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long)
    print("=" * 60)
    print("PROMPT:", repr(PROMPT), f"({idx.size(1)} tokens)")
    print("=" * 60)
    print(f"{'step':>4}  {'cache L':>8}  {'new':>4}  running text")
    print("-" * 60)

    with torch.no_grad():
        logits, caches = model._forward_with_cache(idx, start_pos=0, kv_caches=None)
        print(f"{'pre':>4}  {caches[0][0].size(2):>8}  {'':>4}  {PROMPT}")

        running = PROMPT
        for step in range(1, WALK_STEPS + 1):
            nxt = model._sample_next(logits[:, -1, :], temperature=0.0, top_k=None)
            ch = decode(nxt[0].tolist())
            running += ch
            idx = torch.cat([idx, nxt], dim=1)
            logits, caches = model._forward_with_cache(
                idx[:, -1:], start_pos=idx.size(1) - 1, kv_caches=caches
            )
            shown = running.replace("\n", "\\n")
            print(f"{step:4d}  {caches[0][0].size(2):8d}  {ch!r:>4}  {shown}")

    print()
    print("=" * 60)
    print("FULL GREEDY: cached generate vs old windowed loop")
    print("=" * 60)
    prompt_idx = torch.tensor([encode(PROMPT)], dtype=torch.long)

    def bench(fn):
        fn()
        t0 = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t0

    with torch.no_grad():
        cached, t_c = bench(
            lambda: model.generate(prompt_idx.clone(), max_new_tokens=FULL_NEW_TOKENS, temperature=0.0)
        )
        windowed, t_w = bench(
            lambda: model._generate_windowed(prompt_idx.clone(), FULL_NEW_TOKENS, 0.0, None, None)
        )

    print(f"cached:   {t_c * 1000:.0f} ms")
    print(f"windowed: {t_w * 1000:.0f} ms  ({t_w / t_c:.2f}x)")
    print("tokens identical:", torch.equal(cached, windowed))
    print()
    print("--- cached text ---")
    print(decode(cached[0].tolist()))
    print()
    print("--- windowed text ---")
    print(decode(windowed[0].tolist()))
    print()
    print("=" * 60)
    print("ONE SAMPLED DRAW (temperature=0.8, top_k=40)")
    print("=" * 60)
    torch.manual_seed(0)
    with torch.no_grad():
        sampled = model.generate(prompt_idx.clone(), max_new_tokens=FULL_NEW_TOKENS, temperature=0.8, top_k=40)
    print(decode(sampled[0].tolist()))


if __name__ == "__main__":
    main()
