"""
Build SFT pairs that keep a movie line's meaning and change its tone.

Each row is one Cornell utterance. The prompt shows that modern line. The
response is a short Shakespeare-flavored rewrite of the same meaning. The
loss, via sft_bpe_data.build_example, falls on the rewrite only.

The movie corpus has no such rewrites, so a Claude call writes them. Results
are cached. A dry run is the default; --confirm spends money.

    python style_sft_data.py
    python style_sft_data.py --confirm --n-train 200 --n-eval 40

Train from the Shakespeare checkpoint, not the movie one. Continued
pretraining on raw movie text already erased the register this rewrite is
trying to teach.
"""

import argparse
import json
import os
import random
import re

BASE_DIR = os.path.dirname(__file__)
MOVIE_PATH = os.path.join(BASE_DIR, "data", "movie_dialogue.txt")
OUT_DIR = os.path.join(BASE_DIR, "data", "style_sft")
CACHE_PATH = os.path.join(OUT_DIR, "rewrites.jsonl")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 60
MIN_WORDS = 6
MAX_WORDS = 18

PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}

SYSTEM_PROMPT = (
    "You rewrite one modern movie line into one short line of Shakespeare-flavored "
    "English. Keep the same meaning and the same speech act. Do not add facts, "
    "names, or stage directions. 6 to 16 words. End with . or ? or !. "
    "Output only the rewritten line."
)

SPEAKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 .'\-]{0,40}:$")


def make_prompt(speaker: str, text: str) -> str:
    """The modern line stays in the prompt. PLAY: is the point where the
    rewrite begins, so the model is not asked to continue the movie scene."""
    return f"MOVIE:\n{speaker}: {text}\nPLAY:"


def iter_utterances(path: str):
    speaker = None
    lines: list[str] = []

    def flush():
        if speaker and lines:
            text = " ".join(part.strip() for part in lines if part.strip())
            if text:
                yield_pair = (speaker, text)
                return yield_pair
        return None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                pair = flush()
                if pair:
                    yield pair
                speaker, lines = None, []
                continue
            if SPEAKER_RE.match(line) and not lines:
                speaker = line[:-1]
                continue
            lines.append(line)
        pair = flush()
        if pair:
            yield pair


def eligible(text: str) -> bool:
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if any(ch in text for ch in "{}[]<>"):
        return False
    return True


def select_lines(path: str, n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    pool = []
    seen = set()
    for speaker, text in iter_utterances(path):
        if not eligible(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        pool.append({"speaker": speaker, "text": text, "prompt": make_prompt(speaker, text)})

    rng = random.Random(seed)
    rng.shuffle(pool)
    need = n_train + n_eval
    if len(pool) < need:
        raise RuntimeError(f"only {len(pool)} eligible movie lines, need {need}")
    chosen = pool[:need]
    eval_rows = chosen[:n_eval]
    train_rows = chosen[n_eval:]
    return train_rows, eval_rows


def estimate_cost(n_calls: int, model: str) -> float:
    pricing = PRICING[model]
    input_tokens = n_calls * 180
    output_tokens = n_calls * MAX_OUTPUT_TOKENS
    return (input_tokens / 1e6) * pricing["input"] + (output_tokens / 1e6) * pricing["output"]


def load_cache() -> dict[str, str]:
    if not os.path.exists(CACHE_PATH):
        return {}
    cached = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cached[row["prompt"]] = row["response"]
    return cached


def write_splits(train_rows: list[dict], eval_rows: list[dict], rewrites: dict[str, str]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(path: str, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"prompt": row["prompt"], "response": rewrites[row["prompt"]]}) + "\n")

    dump(os.path.join(OUT_DIR, "train.jsonl"), train_rows)
    dump(os.path.join(OUT_DIR, "eval.jsonl"), eval_rows)


def generate(rows: list[dict], model: str, spend_cap_usd: float) -> dict[str, str]:
    import anthropic

    client = anthropic.Anthropic()
    cached = load_cache()
    pricing = PRICING[model]
    running = 0.0
    os.makedirs(OUT_DIR, exist_ok=True)
    pending = [row for row in rows if row["prompt"] not in cached]
    print(f"{len(cached)} rewrites cached, {len(pending)} to generate")

    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        for i, row in enumerate(pending):
            response = client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.4,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"{row['speaker']}: {row['text']}"}],
            )
            call_cost = (
                response.usage.input_tokens / 1e6 * pricing["input"]
                + response.usage.output_tokens / 1e6 * pricing["output"]
            )
            running += call_cost
            text = response.content[0].text.strip().splitlines()[0].strip()
            cached[row["prompt"]] = text
            f.write(json.dumps({"prompt": row["prompt"], "response": text, "source": row["text"]}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(pending)} generated, ${running:.4f}")
            if running >= spend_cap_usd:
                raise RuntimeError(f"stopped at ${running:.4f}, cap is ${spend_cap_usd:.4f}")
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--spend-cap", type=float, default=1.0)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    train_rows, eval_rows = select_lines(MOVIE_PATH, args.n_train, args.n_eval, args.seed)
    rows = train_rows + eval_rows
    cost = estimate_cost(len(rows), args.model)
    print(f"{len(train_rows)} train, {len(eval_rows)} eval, estimated ${cost:.3f} on {args.model}")
    print("\nexample prompt:\n" + train_rows[0]["prompt"])

    if not args.confirm:
        print("\ndry run. pass --confirm to call the API and write data/style_sft/{train,eval}.jsonl")
        return

    rewrites = generate(rows, args.model, args.spend_cap)
    missing = [row["prompt"] for row in rows if row["prompt"] not in rewrites]
    if missing:
        raise RuntimeError(f"{len(missing)} prompts have no rewrite; re-run --confirm to finish the cache")
    write_splits(train_rows, eval_rows, rewrites)
    print(f"wrote {OUT_DIR}/train.jsonl and eval.jsonl")


if __name__ == "__main__":
    main()
