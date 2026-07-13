"""
Build DPO "gold" preference pairs: instead of the SFT model's own noisy
sample variance (dpo_data.py's approach — a rule-based heuristic ranks K
samples from the SAME weak model), pair genuinely good completions —
generated via the Claude API — against the SFT model's own sampled
completions.

Why: the rule-based-heuristic DPO run (see reports/scratch_nanogpt.md) was
mechanically correct but qualitatively didn't clearly improve over SFT —
classic reward-hacking/proxy-divergence. Root cause: the heuristic's
chosen/rejected gap was often razor-thin (both samples hit EOS, both ended
in punctuation — same tiny model, same failure modes on both sides). A
genuine quality gap (a real completion vs. a 15M-param tiny model's sample)
gives DPO a clearer, more generalizable signal instead of a coin-flip
between two similarly-weak options.

Multiple gold VARIANTS per prompt (--variants-per-prompt) rather than a
single canonical response: with only one gold response per prompt, DPO
risks pulling the policy toward memorizing that one exact phrasing per
speaker rather than learning the general "coherent, non-repetitive" quality
it's meant to stand in for. Prompts are the 287 already validated by
sft_data.py's extraction pipeline — NOT expanded via a naive corpus scan
for more speaker tags: a raw regex pass over tinyshakespeare.txt turns up
~2500 "speakers," but spot-checking shows most are ordinary dialogue lines
that happen to end in ":", not real speaker tags. Widening the prompt pool
that way would poison the gold set with nonsense prompts; scaling via more
variants per already-validated prompt is the safe axis.

Gold completions are deliberately kept in the SAME short, single-clause
register the SFT data already trains on (see SYSTEM_PROMPT) — not
elaborate multi-clause prose. A 15M-param model can't close a gap to
genuinely eloquent writing in a few dozen DPO steps; that big a gap would
just teach surface mimicry, not real quality, and risks the same kind of
instability the too-permissive heuristic caused.

COST CONTROL differs by mode:
- Sequential (default): dry run by default; --confirm required. Aborts
  mid-run using a RUNNING TOTAL of ACTUAL usage from each response, checked
  after every call — the strongest guarantee, since it can stop before the
  next call.
- Batch (--batch, uses the Message Batches API — ~50% cheaper, worth it at
  this larger scale): the whole batch is submitted as one atomic call and
  Anthropic doesn't expose partial results while it's still processing, so
  there's no way to abort mid-batch based on real spend. The only guard is
  a pre-submission estimate check against --spend-cap — a real, if weaker,
  limitation of the batch execution model, not an oversight. At this
  project's scale (low hundreds of short completions, a few cents to
  ~$1 either way) that limitation doesn't matter in practice, but it's
  worth knowing about before pointing this at a much larger prompt set.

Both modes cache results to disk incrementally, so a partial or aborted
run never restarts from zero.

Run:
    export ANTHROPIC_API_KEY=...
    uv sync --extra dev                                   # installs anthropic
    python dpo_gold_data.py                                # dry run, estimate only
    python dpo_gold_data.py --confirm                      # sequential, one call at a time
    python dpo_gold_data.py --confirm --batch --variants-per-prompt 3
"""

import argparse
import json
import os
import random
import time

import torch

from bpe_data import BPETokenizer
from dpo_data import sample_k
from model import TinyGPT

from src.common.generation import resolve_lm_device

BASE_DIR = os.path.dirname(__file__)
SFT_CHECKPOINT = os.path.join(BASE_DIR, "out_sft_bpe", "tinygpt_sft.pt")
SFT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "sft", "train.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "data", "dpo_gold")
GOLD_CACHE_PATH = os.path.join(OUT_DIR, "gold_completions.jsonl")

# Must match out_sft_bpe/'s architecture (gpt_small_bpe.yaml).
MODEL_CFG = dict(block_size=256, n_layer=12, n_head=8, n_embd=128, dropout=0.1)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheap, more than capable for this
MAX_OUTPUT_TOKENS = 60
DEFAULT_K_REJECTED = 3  # SFT samples paired against each gold completion
DEFAULT_VARIANTS_PER_PROMPT = 1
GENERATION_TEMPERATURE = 1.0  # so repeated calls for the same prompt actually vary

# Rough published per-token pricing (USD per million tokens) — approximate,
# used only for the pre-run estimate. Verify current pricing before trusting
# this for a real budget decision; sequential mode's running-total abort
# uses ACTUAL usage from each response, which is the real protection there.
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}

SYSTEM_PROMPT = (
    "You are generating training data for a tiny from-scratch language model "
    "being fine-tuned on Shakespeare's plays. Given a speaker's name or tag, "
    "write ONE short, grammatically clean line of dialogue in period-flavored "
    "English, 5-15 words, ending in proper terminal punctuation (. ? or !). "
    "Do not explain yourself, do not add stage directions, do not repeat the "
    "speaker tag. Output only the line itself, nothing else."
)


def estimate_cost(n_calls: int, model: str) -> float:
    pricing = PRICING[model]
    input_tokens_per_call = 150  # system prompt + short user message, rough
    output_tokens_per_call = MAX_OUTPUT_TOKENS
    total_input = n_calls * input_tokens_per_call
    total_output = n_calls * output_tokens_per_call
    return (total_input / 1e6) * pricing["input"] + (total_output / 1e6) * pricing["output"]


def load_cached_gold() -> dict[str, list[str]]:
    """prompt -> list of gold completion variants generated for it so far."""
    if not os.path.exists(GOLD_CACHE_PATH):
        return {}
    cached: dict[str, list[str]] = {}
    with open(GOLD_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cached.setdefault(row["prompt"], []).append(row["gold_chosen"])
    return cached


def _pending_tasks(prompts: list[str], cached: dict[str, list[str]], variants_per_prompt: int) -> list[str]:
    """One entry per (prompt, missing variant slot) — a flat list of prompts
    to still generate a completion for, repeated once per still-needed variant."""
    tasks = []
    for prompt in prompts:
        have = len(cached.get(prompt, []))
        need = max(0, variants_per_prompt - have)
        tasks.extend([prompt] * need)
    return tasks


def generate_gold_completions_sequential(
    prompts: list[str], model: str, spend_cap_usd: float, variants_per_prompt: int
) -> dict[str, list[str]]:
    import anthropic

    client = anthropic.Anthropic()
    cached = load_cached_gold()
    tasks = _pending_tasks(prompts, cached, variants_per_prompt)
    print(f"{sum(len(v) for v in cached.values())} completions already cached, {len(tasks)} to generate")

    pricing = PRICING[model]
    running_cost = 0.0
    results = {p: list(v) for p, v in cached.items()}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GOLD_CACHE_PATH, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(tasks):
            response = client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=GENERATION_TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Speaker: {prompt}"}],
            )
            call_cost = (
                response.usage.input_tokens / 1e6 * pricing["input"]
                + response.usage.output_tokens / 1e6 * pricing["output"]
            )
            running_cost += call_cost

            text = response.content[0].text.strip()
            results.setdefault(prompt, []).append(text)
            f.write(json.dumps({"prompt": prompt, "gold_chosen": text}) + "\n")
            f.flush()

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(tasks)} generated, running cost so far: ${running_cost:.4f}")

            if running_cost >= spend_cap_usd:
                print(
                    f"\nSTOPPED: running cost ${running_cost:.4f} reached spend cap ${spend_cap_usd:.2f} "
                    f"after {i + 1}/{len(tasks)} calls. {len(tasks) - i - 1} left uncached — "
                    f"re-run to continue from here (cache is incremental, nothing lost)."
                )
                break

    print(f"total spent this run: ${running_cost:.4f}")
    return results


def generate_gold_completions_batch(
    prompts: list[str], model: str, spend_cap_usd: float, variants_per_prompt: int, poll_interval: float = 10.0
) -> dict[str, list[str]]:
    import anthropic

    client = anthropic.Anthropic()
    cached = load_cached_gold()
    tasks = _pending_tasks(prompts, cached, variants_per_prompt)
    print(f"{sum(len(v) for v in cached.values())} completions already cached, {len(tasks)} to generate via batch")

    results = {p: list(v) for p, v in cached.items()}
    if not tasks:
        print("nothing to generate — already at target variant count for every prompt")
        return results

    projected_cost = estimate_cost(len(tasks), model)
    if projected_cost > spend_cap_usd:
        raise RuntimeError(
            f"projected cost ${projected_cost:.4f} for {len(tasks)} calls exceeds spend cap ${spend_cap_usd:.2f} "
            f"— aborting BEFORE submitting the batch (batches can't be partially aborted once running — "
            f"see module docstring). Lower --variants-per-prompt or raise --spend-cap."
        )

    custom_id_to_prompt = {}
    requests = []
    for i, prompt in enumerate(tasks):
        custom_id = f"req_{i}"
        custom_id_to_prompt[custom_id] = prompt
        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": GENERATION_TEMPERATURE,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": f"Speaker: {prompt}"}],
                },
            }
        )

    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} with {len(requests)} requests (est. ${projected_cost:.4f}), polling every {poll_interval:.0f}s...")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        print(f"  status: {batch.processing_status}  counts: {batch.request_counts}")
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)

    pricing = PRICING[model]
    running_cost = 0.0
    n_failed = 0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GOLD_CACHE_PATH, "a", encoding="utf-8") as f:
        for item in client.messages.batches.results(batch.id):
            prompt = custom_id_to_prompt[item.custom_id]
            if item.result.type != "succeeded":
                n_failed += 1
                print(f"  WARNING: request for {prompt!r} did not succeed ({item.result.type}) — skipping")
                continue
            message = item.result.message
            running_cost += (
                message.usage.input_tokens / 1e6 * pricing["input"]
                + message.usage.output_tokens / 1e6 * pricing["output"]
            )
            text = message.content[0].text.strip()
            results.setdefault(prompt, []).append(text)
            f.write(json.dumps({"prompt": prompt, "gold_chosen": text}) + "\n")

    print(f"batch complete. {n_failed} failed requests. total spent: ${running_cost:.4f}")
    return results


def build_pairs(prompts: list[str], gold: dict[str, list[str]], k_rejected: int) -> list[dict]:
    device = resolve_lm_device()
    tokenizer = BPETokenizer()
    sft_model = TinyGPT(vocab_size=tokenizer.vocab_size, **MODEL_CFG).to(device)
    sft_model.load_state_dict(torch.load(SFT_CHECKPOINT, map_location=device))

    pairs = []
    for prompt in prompts:
        gold_variants = gold.get(prompt, [])
        if not gold_variants:
            continue  # not yet generated (spend cap may have stopped early) — skip, don't block on it
        for gold_chosen in gold_variants:
            sft_samples = sample_k(sft_model, tokenizer, prompt, k_rejected, device)
            for s in sft_samples:
                if s["text"].strip() == gold_chosen.strip():
                    continue  # skip degenerate exact ties
                pairs.append({"prompt": prompt, "chosen": gold_chosen, "rejected": s["text"]})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="Actually call the API (default: dry run, estimate only)")
    parser.add_argument("--batch", action="store_true", help="Use the Message Batches API (~50%% cheaper, weaker mid-run abort — see module docstring)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--spend-cap", type=float, default=1.00, help="Hard USD cap")
    parser.add_argument("--k-rejected", type=int, default=DEFAULT_K_REJECTED)
    parser.add_argument("--variants-per-prompt", type=int, default=DEFAULT_VARIANTS_PER_PROMPT)
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between batch status polls (--batch only)")
    args = parser.parse_args()

    with open(SFT_TRAIN_PATH, encoding="utf-8") as f:
        prompts = sorted({json.loads(line)["prompt"] for line in f})

    cached = load_cached_gold()
    tasks = _pending_tasks(prompts, cached, args.variants_per_prompt)
    projected_cost = estimate_cost(len(tasks), args.model)
    print(
        f"{len(prompts)} prompts, target {args.variants_per_prompt} variant(s) each, "
        f"{sum(len(v) for v in cached.values())} completions already cached, {len(tasks)} would be generated"
    )
    print(f"model: {args.model}  mode: {'batch' if args.batch else 'sequential'}  rough pre-run estimate: ${projected_cost:.4f}  spend cap: ${args.spend_cap:.2f}")

    if not args.confirm:
        print("\nDry run only — pass --confirm to actually call the API.")
        return

    if args.batch:
        gold = generate_gold_completions_batch(prompts, args.model, args.spend_cap, args.variants_per_prompt, args.poll_interval)
    else:
        gold = generate_gold_completions_sequential(prompts, args.model, args.spend_cap, args.variants_per_prompt)

    n_completions = sum(len(v) for v in gold.values())
    print(f"gold completions ready: {n_completions} across {len(gold)}/{len(prompts)} prompts -> {GOLD_CACHE_PATH}")

    pairs = build_pairs(prompts, gold, args.k_rejected)
    print(f"built {len(pairs)} pairs from {n_completions} gold completions x up to {args.k_rejected} SFT samples each")

    random.seed(1337)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(OUT_DIR, "eval.jsonl"), "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"train: {len(train_pairs)}  eval: {len(val_pairs)}  -> {OUT_DIR}/")

    print("\n--- 10 random pairs for manual inspection ---")
    for p in random.sample(pairs, min(10, len(pairs))):
        print(f"\nPrompt: {p['prompt']}")
        print(f"  gold (chosen): {p['chosen']!r}")
        print(f"  sft (rejected): {p['rejected']!r}")


if __name__ == "__main__":
    main()
