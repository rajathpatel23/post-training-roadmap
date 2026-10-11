"""Walk cached generate. Default is BPE TinyGPT (out_bpe/).

    python demo_kv_cache.py
    python demo_kv_cache.py --char
"""

import argparse
import json
import time

import torch

from model import TinyGPT

PROMPT = "ROMEO:"
WALK_STEPS = 24
FULL_NEW_TOKENS = 80


def load_char():
    with open("out/tokenizer.json") as f:
        raw = json.load(f)
    stoi = raw["stoi"]
    itos = {int(k): v for k, v in raw["itos"].items()}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    def token_strs(ids: list[int]) -> list[str]:
        return [itos[i] for i in ids]

    return {
        "encode": encode,
        "decode": decode,
        "token_strs": token_strs,
        "vocab_size": len(stoi),
        "eos_id": None,
        "ckpt": "out/tinygpt_shakespeare.pt",
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "label": "char",
    }


def load_bpe():
    from bpe_data import BPETokenizer

    tok = BPETokenizer()

    def token_strs(ids: list[int]) -> list[str]:
        return tok._tok.convert_ids_to_tokens(ids)

    return {
        "encode": tok.encode,
        "decode": tok.decode,
        "token_strs": token_strs,
        "vocab_size": tok.vocab_size,
        "eos_id": tok.eos_id,
        "ckpt": "out_bpe/tinygpt_shakespeare.pt",
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 128,
        "label": "bpe (GPT-2)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", action="store_true", help="use the 4-layer char checkpoint instead")
    args = ap.parse_args()
    cfg = load_char() if args.char else load_bpe()

    model = TinyGPT(
        vocab_size=cfg["vocab_size"],
        block_size=256,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=0.1,
    )
    model.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu", weights_only=True))
    model.eval()

    prompt_ids = cfg["encode"](PROMPT)
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    pieces = cfg["token_strs"](prompt_ids)
    print("=" * 60)
    print(f"tokenizer: {cfg['label']}  ckpt: {cfg['ckpt']}")
    print("PROMPT:", repr(PROMPT), f"({len(prompt_ids)} tokens) -> {pieces}")
    print("=" * 60)
    print(f"{'step':>4}  {'cache L':>8}  {'new piece':<16}  running text")
    print("-" * 60)

    with torch.no_grad():
        logits, caches = model._forward_with_cache(idx, start_pos=0, kv_caches=None)
        print(f"{'pre':>4}  {caches[0][0].size(2):>8}  {'':<16}  {PROMPT}")

        running_ids = list(prompt_ids)
        for step in range(1, WALK_STEPS + 1):
            nxt = model._sample_next(logits[:, -1, :], temperature=0.0, top_k=None)
            tid = int(nxt[0, 0])
            running_ids.append(tid)
            idx = torch.cat([idx, nxt], dim=1)
            logits, caches = model._forward_with_cache(
                idx[:, -1:], start_pos=idx.size(1) - 1, kv_caches=caches
            )
            piece = cfg["token_strs"]([tid])[0]
            shown = cfg["decode"](running_ids).replace("\n", "\\n")
            print(f"{step:4d}  {caches[0][0].size(2):8d}  {piece:<16}  {shown}")

    print()
    print("=" * 60)
    print("FULL GREEDY: cached generate vs old windowed loop")
    print("=" * 60)
    prompt_idx = torch.tensor([prompt_ids], dtype=torch.long)

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
    print(cfg["decode"](cached[0].tolist()))
    print()
    print("--- windowed text ---")
    print(cfg["decode"](windowed[0].tolist()))
    print()
    print("=" * 60)
    print("ONE SAMPLED DRAW (temperature=0.8, top_k=40)")
    print("=" * 60)
    torch.manual_seed(0)
    with torch.no_grad():
        sampled = model.generate(
            prompt_idx.clone(),
            max_new_tokens=FULL_NEW_TOKENS,
            temperature=0.8,
            top_k=40,
            eos_id=cfg["eos_id"],
        )
    print(cfg["decode"](sampled[0].tolist()))


if __name__ == "__main__":
    main()
