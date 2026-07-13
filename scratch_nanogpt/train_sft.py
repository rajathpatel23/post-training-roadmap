"""
SFT: fine-tune the pretrained BPE TinyGPT to produce one complete,
well-formed utterance given a speaker tag, then stop (EOS).

Builds on out_bpe/tinygpt_shakespeare.pt (the pre-movie checkpoint — Run 4
showed continued pretraining on movie dialogue causes real forgetting, so
SFT uses the checkpoint that still has the Shakespeare quality intact).
Loss is masked to response tokens only (prompt conditions the forward pass
but isn't optimized against) — see sft_bpe_data.py's build_example().

Logs a base-vs-SFT qualitative comparison on fixed prompts every eval step,
so the actual behavioral change (not just the loss number) is visible.

Config:
  - init_from: the checkpoint training actually starts from / mutates —
    same meaning as in train.py. On a fresh run this is the original
    pretrained checkpoint; to continue an interrupted SFT run, point this
    at that run's own checkpoint (e.g. out_sft_bpe/tinygpt_sft.pt) instead.
  - reference_checkpoint (optional): the checkpoint used for the qualitative
    "base" comparison samples — defaults to init_from if unset. Set this
    explicitly when init_from points at an already-SFT'd checkpoint (i.e.
    you're continuing an SFT run), so the comparison stays base-pretrained
    vs. latest-SFT rather than silently becoming SFT-vs-SFT.
  - continue_history: if true, continue the step counter/loss history from
    output_dir's existing train_history.json instead of restarting at
    step 0 — independent of init_from, since "which weights to load" and
    "is this a continuation of the same run" are different questions (see
    src/common/checkpointing.py's docstring). Optimizer state (Adam
    moments) is NOT persisted across a continuation — a fresh AdamW warms
    up over the first few steps, a minor transient, not a correctness issue
    for a short extension.

Run:
    python train_sft.py --config configs/train/sft_bpe.yaml
    python train_sft.py --config configs/train/sft_bpe_resume.yaml
"""

import argparse
import dataclasses
import json
import os
import time

import torch

from bpe_data import BPETokenizer
from config import SFTTrainConfig, load_sft_train_config
from model import TinyGPT
from sampling import sample_prompts
from sft_bpe_data import iterate_epoch, load_jsonl, tokenize_examples

from src.common.checkpointing import load_resumable_history
from src.common.generation import resolve_lm_device
from src.common.logging import finish_run, init_run, log_metrics, log_samples
from src.common.plotting import plot_metric_groups

BASE_DIR = os.path.dirname(__file__)

# Fixed speaker prompts for the qualitative callback — same prompts every
# eval so before/after (and epoch-over-epoch) comparison is direct.
EVAL_PROMPTS = ["ROMEO:", "MENENIUS:", "JULIET:", "KING RICHARD III:"]
SAMPLE_MAX_NEW_TOKENS = 80  # SFT responses are one short sentence — no need for pretraining's 120-300 token cap


@torch.no_grad()
def evaluate(model, tokenizer, eval_examples, block_size, batch_size, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for input_ids, labels, attention_mask in iterate_epoch(
        eval_examples, tokenizer, block_size, batch_size, device, shuffle=False
    ):
        _, loss = model(input_ids, targets=labels, key_padding_mask=attention_mask)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main(config_path: str):
    cfg: SFTTrainConfig = load_sft_train_config(config_path)
    assert cfg.model.type == "gpt" and cfg.model.tokenizer == "bpe", (
        "train_sft.py expects a bpe gpt model config (SFT builds on the pre-movie BPE checkpoint)"
    )

    torch.manual_seed(cfg.seed)
    device = resolve_lm_device()
    OUT_DIR = os.path.join(BASE_DIR, cfg.output_dir or "out_sft_bpe")
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = BPETokenizer()
    train_examples = tokenize_examples(tokenizer, load_jsonl(os.path.join(BASE_DIR, cfg.sft_train_path)), cfg.model.block_size)
    eval_examples = tokenize_examples(tokenizer, load_jsonl(os.path.join(BASE_DIR, cfg.sft_eval_path)), cfg.model.block_size)
    print(f"SFT data: {len(train_examples)} train, {len(eval_examples)} eval")

    def build_model():
        return TinyGPT(
            vocab_size=tokenizer.vocab_size,
            block_size=cfg.model.block_size,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            dropout=cfg.model.dropout,
        ).to(device)

    init_from = os.path.join(BASE_DIR, cfg.init_from)
    reference_checkpoint = os.path.join(BASE_DIR, cfg.reference_checkpoint or cfg.init_from)

    # Base-model (pre-SFT) samples always come from reference_checkpoint —
    # defaults to init_from, but stays the true original pretrained
    # checkpoint even when init_from has been pointed at an SFT checkpoint
    # to continue training, so this never silently becomes SFT-vs-SFT.
    base_model = build_model()
    base_model.load_state_dict(torch.load(reference_checkpoint, map_location=device))
    base_samples = sample_prompts(base_model, tokenizer, EVAL_PROMPTS, device, max_new_tokens=SAMPLE_MAX_NEW_TOKENS)
    del base_model

    model = build_model()
    model.load_state_dict(torch.load(init_from, map_location=device))
    print(f"training from: {init_from}")

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    run = init_run(
        {**dataclasses.asdict(cfg), "vocab_size": tokenizer.vocab_size, "n_params": n_params, "device": device},
        output_dir=OUT_DIR,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
    )
    log_samples(run, base_samples, step=0, table_name="base_model_samples")

    history, step = load_resumable_history(OUT_DIR, cfg.continue_history)
    if step:
        print(f"continuing loss history from step {step} ({len(history)} prior eval points)")

    t0 = time.time()
    for epoch in range(cfg.num_epochs):
        for input_ids, labels, attention_mask in iterate_epoch(
            train_examples, tokenizer, cfg.model.block_size, cfg.batch_size, device
        ):
            _, loss = model(input_ids, targets=labels, key_padding_mask=attention_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % cfg.eval_interval == 0:
                eval_loss = evaluate(
                    model, tokenizer, eval_examples, cfg.model.block_size, cfg.batch_size, device
                )
                elapsed = time.time() - t0
                history.append({"step": step, "epoch": epoch, "train_loss": loss.item(), "eval_loss": eval_loss, "elapsed_s": elapsed})
                print(f"epoch {epoch} step {step}: train_loss={loss.item():.4f} eval_loss={eval_loss:.4f} ({elapsed:.0f}s)")

                sft_samples = sample_prompts(model, tokenizer, EVAL_PROMPTS, device, max_new_tokens=SAMPLE_MAX_NEW_TOKENS)
                log_metrics(run, {"train/loss": loss.item(), "eval/loss": eval_loss, "elapsed_s": elapsed}, step=step)
                log_samples(run, sft_samples, step=step, table_name="sft_samples")

        # End-of-epoch eval too, so progress is visible even if eval_interval doesn't divide evenly.
        eval_loss = evaluate(model, tokenizer, eval_examples, cfg.model.block_size, cfg.batch_size, device)
        elapsed = time.time() - t0
        print(f"--- end of epoch {epoch}: eval_loss={eval_loss:.4f} ({elapsed:.0f}s) ---")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "tinygpt_sft.pt"))
    history_path = os.path.join(OUT_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump({"config": dataclasses.asdict(cfg), "history": history, "n_params": n_params}, f, indent=2)

    print(f"done. checkpoint -> {OUT_DIR}/tinygpt_sft.pt")

    if history:
        plot_metric_groups(
            history,
            groups=[{"metrics": ["train_loss", "eval_loss"], "labels": {"train_loss": "train", "eval_loss": "eval"}, "ylabel": "cross-entropy loss", "title": "SFT Loss"}],
            out_path=os.path.join(OUT_DIR, "training_curves.png"),
            title=f"SFT on TinyGPT (bpe) — {n_params:,} params",
        )

    final_samples = sample_prompts(model, tokenizer, EVAL_PROMPTS, device, max_new_tokens=SAMPLE_MAX_NEW_TOKENS)
    log_samples(run, final_samples, step=step, table_name="final_samples")
    finish_run(run)

    print("\n--- base model (pre-SFT) vs SFT, same prompts ---")
    for base, sft in zip(base_samples, final_samples, strict=True):
        print(f"\nPrompt: {base['prompt']}")
        print(f"  base: {base['output']!r}")
        print(f"  sft:  {sft['output']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/sft_bpe.yaml")
    args = parser.parse_args()
    main(args.config)
