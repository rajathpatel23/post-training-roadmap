"""
Pretrain TinyBERT (MLM + NSP) on tiny-shakespeare, char-level, from random init.

Same conventions as train.py: config-driven architecture/hyperparams, shared
src/common/ logging + plotting (not duplicated per-script).

Run:
    python train_bert.py --config configs/train/bert_small.yaml
"""

import argparse
import dataclasses
import json
import math
import os
import time

import torch

from bert_data import CLS_ID, MASK_ID, build_example, get_batch, load_lines
from config import BERTTrainConfig, load_bert_train_config
from model import TinyBERT

from src.common.generation import resolve_lm_device
from src.common.logging import finish_run, init_run, log_metrics, log_samples
from src.common.plotting import plot_metric_groups

BASE_DIR = os.path.dirname(__file__)

# Fixed masked-sentence probes for the callback — same input every eval, so
# improvement in the model's masked-token predictions is directly comparable
# across steps. "_" marks the position we mask and check the top prediction for.
PROBE_SENTENCES = ["ROMEO: _hat light through yonder window breaks", "To be, or not to _e"]

# Fixed NSP probes — one plausible continuation (label 0, IsNext), one
# mismatched pair (label 1, NotNext). Same pairs every eval so the model's
# predicted confidence is directly comparable across steps.
NSP_PROBE_PAIRS = [
    ("ROMEO:", "But soft, what light through yonder window breaks?", 0),
    ("ROMEO:", "To be, or not to be, that is the question:", 1),
]


@torch.no_grad()
def estimate_loss(model, tokenizer, lines_train, lines_val, cfg: BERTTrainConfig, device):
    model.eval()
    out = {}
    for split, lines in (("train", lines_train), ("val", lines_val)):
        mlm_losses = torch.zeros(cfg.eval_iters)
        nsp_losses = torch.zeros(cfg.eval_iters)
        nsp_correct = 0
        nsp_total = 0
        for k in range(cfg.eval_iters):
            input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels = get_batch(
                tokenizer, lines, cfg.model.block_size, cfg.batch_size, cfg.mlm_probability, device
            )
            out_dict = model(input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
            mlm_losses[k] = out_dict["mlm_loss"].item()
            nsp_losses[k] = out_dict["nsp_loss"].item()
            preds = out_dict["nsp_logits"].argmax(dim=-1)
            nsp_correct += (preds == nsp_labels).sum().item()
            nsp_total += nsp_labels.numel()
        out[f"{split}_mlm_loss"] = mlm_losses.mean().item()
        out[f"{split}_nsp_loss"] = nsp_losses.mean().item()
        out[f"{split}_nsp_acc"] = nsp_correct / nsp_total
    model.train()
    return out


@torch.no_grad()
def probe_masked_predictions(model, tokenizer, sentences, block_size, device):
    """Callback: mask the char at '_' in each probe sentence, show the
    model's top-1 prediction there. Qualitative signal alongside MLM loss."""
    model.eval()
    rows = []
    for s in sentences:
        mask_pos_char = s.index("_")
        clean = s.replace("_", "")
        ids = tokenizer.encode(clean)
        mask_pos = mask_pos_char + 1  # +1 for the [CLS] prefix

        token_ids = [CLS_ID] + ids
        token_ids = token_ids[:block_size]
        if mask_pos >= len(token_ids):
            continue
        original_char = clean[mask_pos_char]
        token_ids[mask_pos] = MASK_ID

        pad_len = block_size - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        segment_ids = [0] * block_size
        token_ids = token_ids + [0] * pad_len

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        seg = torch.tensor([segment_ids], dtype=torch.long, device=device)
        mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

        out_dict = model(input_ids, seg, mask)
        pred_id = out_dict["mlm_logits"][0, mask_pos].argmax().item()
        pred_char = tokenizer.decode([pred_id])

        rows.append({"sentence": s, "true_char": original_char, "predicted_char": pred_char})
    model.train()
    return rows


@torch.no_grad()
def probe_nsp_predictions(model, tokenizer, pairs, block_size, device):
    """Callback: run fixed (seg_a, seg_b, true_label) pairs through the model,
    show predicted label + confidence. Qualitative signal alongside NSP
    accuracy — no MLM masking applied here (mlm_probability=0.0) so the
    prediction reflects NSP judgment specifically, not masked-token noise."""
    model.eval()
    rows = []
    label_names = {0: "IsNext", 1: "NotNext"}
    for seg_a, seg_b, true_label in pairs:
        example = build_example(tokenizer, seg_a, seg_b, true_label, block_size, mlm_probability=0.0)

        input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
        seg = torch.tensor([example["segment_ids"]], dtype=torch.long, device=device)
        mask = torch.tensor([example["attention_mask"]], dtype=torch.long, device=device)

        out_dict = model(input_ids, seg, mask)
        probs = torch.softmax(out_dict["nsp_logits"][0], dim=-1)
        pred_label = probs.argmax().item()

        rows.append(
            {
                "seg_a": seg_a,
                "seg_b": seg_b,
                "true_label": label_names[true_label],
                "predicted_label": label_names[pred_label],
                "confidence": f"{probs[pred_label].item():.3f}",
            }
        )
    model.train()
    return rows


def main(config_path: str):
    cfg = load_bert_train_config(config_path)
    assert cfg.model.type == "bert", f"train_bert.py expects a bert model config, got {cfg.model.type}"

    OUT_DIR = os.path.join(BASE_DIR, cfg.output_dir or "out_bert")

    torch.manual_seed(cfg.seed)
    device = resolve_lm_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer, lines_train, lines_val = load_lines()
    print(f"vocab_size={tokenizer.vocab_size}  train_lines={len(lines_train)}  val_lines={len(lines_val)}  device={device}")

    model = TinyBERT(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    run = init_run(
        {**dataclasses.asdict(cfg), "vocab_size": tokenizer.vocab_size, "n_params": n_params, "device": device},
        output_dir=OUT_DIR,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
    )

    history = []
    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, tokenizer, lines_train, lines_val, cfg, device)
            elapsed = time.time() - t0
            val_mlm_ppl = math.exp(min(losses["val_mlm_loss"], 20.0))
            history.append({"step": it, **losses, "elapsed_s": elapsed, "val_mlm_ppl": val_mlm_ppl})
            print(
                f"step {it}: train_mlm={losses['train_mlm_loss']:.4f} val_mlm={losses['val_mlm_loss']:.4f} "
                f"val_mlm_ppl={val_mlm_ppl:.2f} val_nsp_acc={losses['val_nsp_acc']:.3f} ({elapsed:.0f}s)"
            )

            probes = probe_masked_predictions(model, tokenizer, PROBE_SENTENCES, cfg.model.block_size, device)
            nsp_probes = probe_nsp_predictions(model, tokenizer, NSP_PROBE_PAIRS, cfg.model.block_size, device)

            log_metrics(
                run,
                {
                    "train/mlm_loss": losses["train_mlm_loss"],
                    "val/mlm_loss": losses["val_mlm_loss"],
                    "train/nsp_loss": losses["train_nsp_loss"],
                    "val/nsp_loss": losses["val_nsp_loss"],
                    "val/mlm_perplexity": val_mlm_ppl,
                    "train/nsp_acc": losses["train_nsp_acc"],
                    "val/nsp_acc": losses["val_nsp_acc"],
                    "elapsed_s": elapsed,
                },
                step=it,
            )
            log_samples(
                run, probes, step=it, table_name="mask_probes", columns=["sentence", "true_char", "predicted_char"]
            )
            log_samples(
                run,
                nsp_probes,
                step=it,
                table_name="nsp_probes",
                columns=["seg_a", "seg_b", "true_label", "predicted_label", "confidence"],
            )

        if it == cfg.max_iters:
            break

        input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels = get_batch(
            tokenizer, lines_train, cfg.model.block_size, cfg.batch_size, cfg.mlm_probability, device
        )
        out_dict = model(input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
        loss = out_dict["mlm_loss"] + out_dict["nsp_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "tinybert_shakespeare.pt"))
    history_path = os.path.join(OUT_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump({"config": dataclasses.asdict(cfg), "history": history, "n_params": n_params}, f, indent=2)

    print(f"done. checkpoint -> {OUT_DIR}/tinybert_shakespeare.pt")

    plot_metric_groups(
        history,
        groups=[
            {
                "metrics": ["train_mlm_loss", "val_mlm_loss"],
                "labels": {"train_mlm_loss": "train", "val_mlm_loss": "val"},
                "ylabel": "MLM cross-entropy loss",
                "title": "MLM Loss",
            },
            {"metrics": ["val_mlm_ppl"], "ylabel": "perplexity (masked positions)", "title": "MLM Perplexity (val)"},
            {
                "metrics": ["train_nsp_acc", "val_nsp_acc"],
                "labels": {"train_nsp_acc": "train", "val_nsp_acc": "val"},
                "ylabel": "accuracy",
                "title": "NSP Accuracy",
                "hline": 0.5,
                "hline_label": "chance",
                "ylim": (0, 1),
            },
        ],
        out_path=os.path.join(OUT_DIR, "training_curves.png"),
        title=f"TinyBERT (MLM+NSP) on tiny-shakespeare — {n_params:,} params",
    )

    finish_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/bert_small.yaml")
    args = parser.parse_args()
    main(args.config)
