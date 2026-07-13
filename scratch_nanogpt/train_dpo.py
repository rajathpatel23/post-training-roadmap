"""
DPO: preference-optimize the SFT policy using (prompt, chosen, rejected)
triples built by dpo_data.py from the SFT model's own sample variance,
scored with a rule-based heuristic (no LLM judge — see dpo_data.py).

Two model copies: a trainable policy and a frozen reference, both
initialized from the same SFT checkpoint — standard DPO setup. The loss
pushes the policy's preference margin (chosen vs. rejected, relative to
what the reference already assigns) positive, without letting the policy
drift arbitrarily far from the reference.

Logs a three-way qualitative comparison (base pretrained -> SFT -> DPO) on
fixed prompts every eval step — the actual learning trajectory across all
three post-training stages on the same prompts, not just SFT->DPO — plus
DPO-specific diagnostics beyond the loss number: reward margin (chosen -
rejected) and preference accuracy (% of pairs the policy now scores chosen
above rejected). The loss value alone would just look like "another
loss going down," identical in shape to SFT's curve; margin/accuracy are
what actually show the DPO-specific dynamic.

Run:
    python train_dpo.py --config configs/train/dpo_bpe.yaml
"""

import argparse
import dataclasses
import json
import os
import time

import torch
import torch.nn.functional as F

from bpe_data import BPETokenizer
from config import DPOTrainConfig, load_dpo_train_config
from dpo_bpe_data import iterate_epoch, load_jsonl, tokenize_examples
from model import TinyGPT
from sampling import sample_prompts, sample_prompts_multi
from sft_bpe_data import IGNORE_INDEX

from src.common.generation import resolve_lm_device
from src.common.logging import finish_run, init_run, log_metrics, log_samples
from src.common.plotting import plot_metric_groups

BASE_DIR = os.path.dirname(__file__)

EVAL_PROMPTS = ["ROMEO:", "MENENIUS:", "JULIET:", "KING RICHARD III:"]
SAMPLE_MAX_NEW_TOKENS = 80

# A single stochastic draw per prompt can't tell genuine policy collapse
# apart from an unlucky low-probability sample (see the gold-dataset run's
# diagnosis in reports/scratch_nanogpt.md) — draw this many per prompt
# instead, both during training and for the final checkpoint.
QUALITATIVE_SAMPLES_K = 3


def sequence_logprobs(model, input_ids, labels, attention_mask):
    """Sum of log p(token | context) over response positions only (labels
    != IGNORE_INDEX) — the per-sequence quantity DPO's loss compares
    between chosen/rejected and policy/reference."""
    logits, _ = model(input_ids, key_padding_mask=attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != IGNORE_INDEX
    safe_labels = labels.clone()
    safe_labels[~mask] = 0  # placeholder id — zeroed out below, value never used
    token_logprobs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float):
    policy_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()

    chosen_reward = (beta * (policy_chosen - ref_chosen)).detach()
    rejected_reward = (beta * (policy_rejected - ref_rejected)).detach()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, chosen_reward.mean(), rejected_reward.mean(), accuracy


@torch.no_grad()
def evaluate(policy, reference, tokenizer, eval_examples, block_size, batch_size, device, beta):
    policy.eval()
    total_loss, total_acc, total_margin, n_batches = 0.0, 0.0, 0.0, 0
    for chosen_batch, rejected_batch in iterate_epoch(
        eval_examples, tokenizer, block_size, batch_size, device, shuffle=False
    ):
        c_ids, c_labels, c_mask = chosen_batch
        r_ids, r_labels, r_mask = rejected_batch

        policy_chosen = sequence_logprobs(policy, c_ids, c_labels, c_mask)
        policy_rejected = sequence_logprobs(policy, r_ids, r_labels, r_mask)
        ref_chosen = sequence_logprobs(reference, c_ids, c_labels, c_mask)
        ref_rejected = sequence_logprobs(reference, r_ids, r_labels, r_mask)

        loss, chosen_reward, rejected_reward, accuracy = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta
        )
        total_loss += loss.item()
        total_acc += accuracy.item()
        total_margin += (chosen_reward - rejected_reward).item()
        n_batches += 1

    policy.train()
    n = max(n_batches, 1)
    return total_loss / n, total_acc / n, total_margin / n


def main(config_path: str):
    cfg: DPOTrainConfig = load_dpo_train_config(config_path)
    assert cfg.model.type == "gpt" and cfg.model.tokenizer == "bpe", "train_dpo.py expects a bpe gpt model config"

    torch.manual_seed(cfg.seed)
    device = resolve_lm_device()
    OUT_DIR = os.path.join(BASE_DIR, cfg.output_dir or "out_dpo_bpe")
    os.makedirs(OUT_DIR, exist_ok=True)
    CHECKPOINTS_DIR = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    tokenizer = BPETokenizer()
    train_examples = tokenize_examples(
        tokenizer, load_jsonl(os.path.join(BASE_DIR, cfg.dpo_train_path)), cfg.model.block_size
    )
    eval_examples = tokenize_examples(
        tokenizer, load_jsonl(os.path.join(BASE_DIR, cfg.dpo_eval_path)), cfg.model.block_size
    )
    print(f"DPO data: {len(train_examples)} train pairs, {len(eval_examples)} eval pairs")

    def build_model():
        return TinyGPT(
            vocab_size=tokenizer.vocab_size,
            block_size=cfg.model.block_size,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            dropout=cfg.model.dropout,
        ).to(device)

    sft_checkpoint = os.path.join(BASE_DIR, cfg.sft_checkpoint)
    pretrained_checkpoint = os.path.join(BASE_DIR, cfg.pretrained_checkpoint)

    # Three-way qualitative comparison: base pretrained -> SFT -> DPO (in
    # progress) on the same prompts, captured once before any DPO step.
    pretrained_model = build_model()
    pretrained_model.load_state_dict(torch.load(pretrained_checkpoint, map_location=device))
    pretrained_samples = sample_prompts(
        pretrained_model, tokenizer, EVAL_PROMPTS, device, max_new_tokens=SAMPLE_MAX_NEW_TOKENS
    )
    del pretrained_model

    sft_model_for_samples = build_model()
    sft_model_for_samples.load_state_dict(torch.load(sft_checkpoint, map_location=device))
    sft_samples = sample_prompts(
        sft_model_for_samples, tokenizer, EVAL_PROMPTS, device, max_new_tokens=SAMPLE_MAX_NEW_TOKENS
    )
    del sft_model_for_samples

    # Reference — frozen for the entire run, never updated, eval mode
    # permanently (no dropout noise in the reference signal).
    reference = build_model()
    reference.load_state_dict(torch.load(sft_checkpoint, map_location=device))
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    # Policy — the model actually being trained, also init'd from the SFT checkpoint.
    policy = build_model()
    policy.load_state_dict(torch.load(sft_checkpoint, map_location=device))

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"training from: {sft_checkpoint}  (beta={cfg.beta})")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

    run = init_run(
        {**dataclasses.asdict(cfg), "vocab_size": tokenizer.vocab_size, "n_params": n_params, "device": device},
        output_dir=OUT_DIR,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
    )
    log_samples(run, pretrained_samples, step=0, table_name="pretrained_samples")
    log_samples(run, sft_samples, step=0, table_name="sft_samples")

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(cfg.num_epochs):
        for chosen_batch, rejected_batch in iterate_epoch(
            train_examples, tokenizer, cfg.model.block_size, cfg.batch_size, device
        ):
            c_ids, c_labels, c_mask = chosen_batch
            r_ids, r_labels, r_mask = rejected_batch

            policy_chosen = sequence_logprobs(policy, c_ids, c_labels, c_mask)
            policy_rejected = sequence_logprobs(policy, r_ids, r_labels, r_mask)
            with torch.no_grad():
                ref_chosen = sequence_logprobs(reference, c_ids, c_labels, c_mask)
                ref_rejected = sequence_logprobs(reference, r_ids, r_labels, r_mask)

            loss, chosen_reward, rejected_reward, accuracy = dpo_loss(
                policy_chosen, policy_rejected, ref_chosen, ref_rejected, cfg.beta
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % cfg.eval_interval == 0:
                eval_loss, eval_acc, eval_margin = evaluate(
                    policy, reference, tokenizer, eval_examples, cfg.model.block_size, cfg.batch_size, device, cfg.beta
                )
                elapsed = time.time() - t0
                train_margin = (chosen_reward - rejected_reward).item()
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        "eval_loss": eval_loss,
                        "train_margin": train_margin,
                        "eval_margin": eval_margin,
                        "train_acc": accuracy.item(),
                        "eval_acc": eval_acc,
                        "elapsed_s": elapsed,
                    }
                )
                print(
                    f"epoch {epoch} step {step}: train_loss={loss.item():.4f} eval_loss={eval_loss:.4f} "
                    f"eval_margin={eval_margin:+.3f} eval_acc={eval_acc:.3f} ({elapsed:.0f}s)"
                )

                dpo_samples = sample_prompts_multi(
                    policy, tokenizer, EVAL_PROMPTS, device, k=QUALITATIVE_SAMPLES_K, max_new_tokens=SAMPLE_MAX_NEW_TOKENS
                )
                log_metrics(
                    run,
                    {
                        "train/loss": loss.item(),
                        "eval/loss": eval_loss,
                        "train/reward_margin": train_margin,
                        "eval/reward_margin": eval_margin,
                        "train/accuracy": accuracy.item(),
                        "eval/accuracy": eval_acc,
                        "elapsed_s": elapsed,
                    },
                    step=step,
                )
                log_samples(run, dpo_samples, step=step, table_name="dpo_samples")

                # Intermediate checkpoint — so a good stopping point found
                # after the fact (from the curves or the sample tables) can
                # actually be recovered, not just inferred.
                torch.save(policy.state_dict(), os.path.join(CHECKPOINTS_DIR, f"step_{step}.pt"))

        eval_loss, eval_acc, eval_margin = evaluate(
            policy, reference, tokenizer, eval_examples, cfg.model.block_size, cfg.batch_size, device, cfg.beta
        )
        elapsed = time.time() - t0
        print(f"--- end of epoch {epoch}: eval_loss={eval_loss:.4f} eval_acc={eval_acc:.3f} ({elapsed:.0f}s) ---")

    torch.save(policy.state_dict(), os.path.join(OUT_DIR, "tinygpt_dpo.pt"))
    history_path = os.path.join(OUT_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump({"config": dataclasses.asdict(cfg), "history": history, "n_params": n_params}, f, indent=2)

    print(f"done. checkpoint -> {OUT_DIR}/tinygpt_dpo.pt")

    if history:
        plot_metric_groups(
            history,
            groups=[
                {
                    "metrics": ["train_loss", "eval_loss"],
                    "labels": {"train_loss": "train", "eval_loss": "eval"},
                    "ylabel": "DPO loss",
                    "title": "Loss",
                },
                {
                    "metrics": ["train_margin", "eval_margin"],
                    "labels": {"train_margin": "train", "eval_margin": "eval"},
                    "ylabel": "reward margin (chosen - rejected)",
                    "title": "Reward Margin",
                    "hline": 0.0,
                    "hline_label": "no preference",
                },
                {
                    "metrics": ["train_acc", "eval_acc"],
                    "labels": {"train_acc": "train", "eval_acc": "eval"},
                    "ylabel": "accuracy",
                    "title": "Preference Accuracy",
                    "hline": 0.5,
                    "hline_label": "chance",
                    "ylim": (0, 1),
                },
            ],
            out_path=os.path.join(OUT_DIR, "training_curves.png"),
            title=f"DPO on TinyGPT (bpe) — {n_params:,} params, beta={cfg.beta}",
        )

    final_samples = sample_prompts(policy, tokenizer, EVAL_PROMPTS, device, max_new_tokens=SAMPLE_MAX_NEW_TOKENS)
    final_samples_multi = sample_prompts_multi(
        policy, tokenizer, EVAL_PROMPTS, device, k=QUALITATIVE_SAMPLES_K, max_new_tokens=SAMPLE_MAX_NEW_TOKENS
    )
    log_samples(run, final_samples, step=step, table_name="final_samples")
    log_samples(run, final_samples_multi, step=step, table_name="final_samples_multi")
    finish_run(run)

    print(f"\n--- final checkpoint, {QUALITATIVE_SAMPLES_K} draws per prompt (checking for noise vs. real collapse) ---")
    for p in EVAL_PROMPTS:
        print(f"\nPrompt: {p}")
        for row in final_samples_multi:
            if row["prompt"] == p:
                print(f"  draw {row['draw']}: {row['output']!r}")

    print("\n--- base pretrained vs SFT vs DPO, same prompts (single draw each) ---")
    for pre, sft, dpo in zip(pretrained_samples, sft_samples, final_samples, strict=True):
        print(f"\nPrompt: {pre['prompt']}")
        print(f"  pretrained: {pre['output']!r}")
        print(f"  sft:        {sft['output']!r}")
        print(f"  dpo:        {dpo['output']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/dpo_bpe.yaml")
    args = parser.parse_args()
    main(args.config)
