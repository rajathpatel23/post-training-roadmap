"""
Pretrain (or continue-pretrain) TinyGPT, from random init or from an
existing checkpoint.

Tokenizer is a config flag, not a code fork: `model.tokenizer: char` (data.py,
corpus-locked to tiny-shakespeare's exact character set) or `model.tokenizer:
bpe` (bpe_data.py, GPT-2/GPT-3's original 50,257-token byte-level BPE vocab —
covers any standard English text, so it's the one that supports training on a
different corpus). Config fields (see config.py's GPTTrainConfig):
  - data_path: override the training corpus (bpe tokenizer only — char's
    vocab can't handle characters outside tiny-shakespeare).
  - init_from: path to an existing .pt checkpoint to continue training from,
    instead of random init. Architecture in this config must match the
    checkpoint's (same n_layer/n_head/n_embd/vocab) or load_state_dict fails.
  - continue_history: if true AND init_from is set, continue the step
    counter/loss history from output_dir's existing train_history.json
    instead of restarting at step 0. Independent of init_from — loading a
    checkpoint doesn't by itself imply "this is a resumed run of the same
    stage" (e.g. Run 4 loads out_bpe/'s checkpoint but is a new training
    stage — continued pretraining on a different corpus — so it correctly
    defaults to a fresh history, not a continued one).
  - secondary_eval_path: score val loss on a second, fixed corpus every eval
    step too (bpe only) — the forgetting check for continued pretraining:
    did the model get worse at the *original* domain while learning the new
    one? Uses the same tokenizer instance, no extra download/reload.
  - output_dir: override the auto out/ | out_bpe/ naming.

Architecture and training hyperparams come entirely from the YAML config —
to try a different size/shape, point at a different config file, don't edit
this script.

Logging/plotting go through src/common/ (shared across every project in this
repo, not just this one) — init_run/log_metrics/log_samples/finish_run for
W&B, plot_metric_groups for the local PNG. Falls back to local-only if W&B
isn't configured (set WANDB_API_KEY / run `wandb login` to enable).

Run:
    python train.py --config configs/train/gpt_small.yaml               # char, from scratch
    python train.py --config configs/train/gpt_small_bpe.yaml           # BPE, from scratch
    python train.py --config configs/train/gpt_bpe_movie_continued.yaml # BPE, continued from Shakespeare checkpoint
"""

import argparse
import dataclasses
import json
import math
import os
import time

import torch

from config import GPTTrainConfig, load_gpt_train_config
from data import get_batch
from model import TinyGPT
from sampling import sample_prompts

from src.common.checkpointing import load_resumable_history
from src.common.generation import resolve_lm_device
from src.common.logging import finish_run, init_run, log_metrics, log_samples
from src.common.plotting import plot_metric_groups

BASE_DIR = os.path.dirname(__file__)

# Fixed seed prompts for the generation callback — same prompts every eval so
# improvement over training steps is directly comparable. Plain English +
# basic punctuation, valid under either tokenizer (char or BPE).
SAMPLE_PROMPTS = ["\n", "ROMEO:", "To be, or not"]


def load_tokenizer_and_data(tokenizer_type: str, data_path: str | None = None):
    if tokenizer_type == "bpe":
        from bpe_data import SHAKESPEARE_PATH, load_data as load_data_bpe

        return load_data_bpe(data_path=data_path or SHAKESPEARE_PATH)
    if data_path is not None:
        raise ValueError(
            "data_path override isn't supported for the char tokenizer — its vocab is "
            "locked to the corpus it was built from (tiny-shakespeare). Use tokenizer: bpe "
            "in the model config for a different corpus."
        )
    import data

    return data.load_data()


def load_secondary_eval_data(tokenizer_type: str, tokenizer, secondary_eval_path: str):
    if tokenizer_type != "bpe":
        raise ValueError("secondary_eval_path (forgetting check) is only supported for tokenizer: bpe.")
    from bpe_data import encode_corpus

    _, val_data = encode_corpus(tokenizer, secondary_eval_path)
    return val_data


@torch.no_grad()
def estimate_loss_on(model, data_tensor, block_size, batch_size, eval_iters, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data_tensor, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: GPTTrainConfig, device):
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        out[split] = estimate_loss_on(model, data, cfg.model.block_size, cfg.batch_size, cfg.eval_iters, device)
    return out


def main(config_path: str):
    cfg = load_gpt_train_config(config_path)
    assert cfg.model.type == "gpt", f"train.py expects a gpt model config, got {cfg.model.type}"
    tokenizer_type = cfg.model.tokenizer

    out_dir_name = cfg.output_dir or ("out" if tokenizer_type == "char" else f"out_{tokenizer_type}")
    OUT_DIR = os.path.join(BASE_DIR, out_dir_name)

    torch.manual_seed(cfg.seed)
    device = resolve_lm_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer, train_data, val_data = load_tokenizer_and_data(tokenizer_type, cfg.data_path)
    print(
        f"tokenizer={tokenizer_type}  vocab_size={tokenizer.vocab_size}  "
        f"train_tokens={len(train_data)}  val_tokens={len(val_data)}  device={device}"
    )

    secondary_val_data = None
    secondary_name = None
    if cfg.secondary_eval_path:
        secondary_val_data = load_secondary_eval_data(tokenizer_type, tokenizer, cfg.secondary_eval_path)
        secondary_name = os.path.splitext(os.path.basename(cfg.secondary_eval_path))[0]
        print(f"forgetting check enabled: also scoring val loss on {cfg.secondary_eval_path}")

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
    ).to(device)

    if cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location=device))
        print(f"continuing from checkpoint: {cfg.init_from}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    run = init_run(
        {**dataclasses.asdict(cfg), "vocab_size": tokenizer.vocab_size, "n_params": n_params, "device": device},
        output_dir=OUT_DIR,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
    )

    history, start_step = load_resumable_history(OUT_DIR, cfg.continue_history)
    t0 = time.time()
    for it in range(start_step, start_step + cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            elapsed = time.time() - t0
            train_ppl = math.exp(min(losses["train"], 20.0))
            val_ppl = math.exp(min(losses["val"], 20.0))
            entry = {"step": it, **losses, "elapsed_s": elapsed, "train_ppl": train_ppl, "val_ppl": val_ppl}

            metrics = {
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "train/perplexity": train_ppl,
                "val/perplexity": val_ppl,
                "elapsed_s": elapsed,
            }
            log_msg = f"step {it}: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}"

            if secondary_val_data is not None:
                secondary_loss = estimate_loss_on(
                    model, secondary_val_data, cfg.model.block_size, cfg.batch_size, cfg.eval_iters, device
                )
                entry[f"{secondary_name}_val_loss"] = secondary_loss
                metrics[f"{secondary_name}/val_loss"] = secondary_loss
                log_msg += f"  {secondary_name}_val_loss={secondary_loss:.4f}"

            history.append(entry)
            print(log_msg + f" val_ppl={val_ppl:.2f} ({elapsed:.0f}s)")

            samples = sample_prompts(model, tokenizer, SAMPLE_PROMPTS, device)

            log_metrics(run, metrics, step=it)
            log_samples(run, samples, step=it, table_name="samples")

        if it == start_step + cfg.max_iters:
            break

        x, y = get_batch(train_data, cfg.model.block_size, cfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "tinygpt_shakespeare.pt"))
    if tokenizer_type == "char":
        # BPE tokenizer isn't saved — it's just the pretrained GPT-2 vocab,
        # reloadable via GPT2TokenizerFast.from_pretrained("gpt2") anytime.
        with open(os.path.join(OUT_DIR, "tokenizer.json"), "w") as f:
            json.dump({"stoi": tokenizer.stoi, "itos": {str(k): v for k, v in tokenizer.itos.items()}}, f)
    history_path = os.path.join(OUT_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump({"config": dataclasses.asdict(cfg), "history": history, "n_params": n_params}, f, indent=2)

    print(f"done. checkpoint -> {OUT_DIR}/tinygpt_shakespeare.pt")

    loss_metrics = ["train", "val"]
    loss_labels = {"train": "train", "val": "val"}
    if secondary_name is not None:
        loss_metrics.append(f"{secondary_name}_val_loss")
        loss_labels[f"{secondary_name}_val_loss"] = f"{secondary_name} (val)"

    plot_metric_groups(
        history,
        groups=[
            {"metrics": loss_metrics, "labels": loss_labels, "ylabel": "cross-entropy loss", "title": "Loss"},
            {"metrics": ["train_ppl", "val_ppl"], "labels": {"train_ppl": "train", "val_ppl": "val"}, "ylabel": "perplexity", "title": "Perplexity"},
        ],
        out_path=os.path.join(OUT_DIR, "training_curves.png"),
        title=f"TinyGPT ({tokenizer_type}) — {n_params:,} params",
    )

    finish_run(run)

    # Final sanity sample, seeded with a newline (id 0 is [EOS] for char — don't seed with that).
    final_sample = sample_prompts(model, tokenizer, ["\n"], device, max_new_tokens=300)[0]
    print("\n--- final sample ---")
    print(final_sample["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/gpt_small.yaml")
    args = parser.parse_args()
    main(args.config)
