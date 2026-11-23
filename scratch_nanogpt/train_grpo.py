"""
GRPO: RL-style optimization branched from the SFT checkpoint, using the
same rule-based well-formedness heuristic (score_completion, from
dpo_data.py) as a verifiable reward — no learned reward model, no LLM judge.

Why branch from SFT (not stack on DPO): the blog arc treats DPO and GRPO as
two ways to improve an SFT model — preferences offline vs live reward on
policy samples — so both should share the same parent checkpoint.

Group Relative Policy Optimization, in the form used here: for each prompt,
sample a *group* of K completions live from the current policy, score each
with the heuristic, and use the group's own mean/std as the baseline
(advantage = (reward - group_mean) / group_std) instead of a learned value
function/critic. That's the "relative" in GRPO — no critic to train, no
separate value network, at the cost of needing several live samples per
prompt per step instead of one.

Central mechanical difference from DPO: DPO trains over a *static* preference
dataset built once (offline, many epochs over fixed pairs). GRPO has no such
dataset — every step samples fresh completions from the current policy,
scores them, and updates. On-policy: slower per step (generation dominates),
but the reward always reflects the policy being trained.

The policy-gradient step reuses sequence_logprobs with reduction="mean"
(length-normalized) weighted by group-relative advantage. DPO keeps the
default sum — same tokens, different aggregation, so long replies don't
dominate GRPO's gradient. KL is the mean log-prob gap vs frozen SFT.
Simplification vs. textbook GRPO: no k3 estimator; mu=1 so no π_new/π_old
clip yet.

Carry-forward from the DPO debugging arc: sequence-summed / advantage-
weighted losses need a lower lr than SFT's per-token-mean loss. Watch
gradient magnitude vs. loss shape early before trusting the lr.

Run:
    python train_grpo.py --config configs/train/grpo_bpe.yaml
"""

import argparse
import dataclasses
import json
import os
import time

import torch

from bpe_data import BPETokenizer
from config import GRPOTrainConfig, load_grpo_train_config
from dpo_data import score_completion
from model import TinyGPT
from sampling import sample_prompts
from sequence_logprobs import sequence_logprobs
from sft_bpe_data import build_example, collate

from src.common.generation import resolve_lm_device
from src.common.logging import finish_run, init_run, log_metrics, log_samples
from src.common.plotting import plot_metric_groups

BASE_DIR = os.path.dirname(__file__)

EVAL_PROMPTS = ["ROMEO:", "MENENIUS:", "JULIET:", "KING RICHARD III:"]
GROUP_ADV_EPS = 1e-4  # guards the group std denominator when all K completions score identically


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return sorted({json.loads(line)["prompt"] for line in f})


@torch.no_grad()
def sample_group(model, tokenizer, prompt: str, k: int, device: str, max_new_tokens: int):
    """Sample K completions for one prompt from the current policy — same
    shape as dpo_data.py's sample_k, reused here as the live rollout step
    instead of an offline data-prep step."""
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    completions = []
    for _ in range(k):
        out_ids = model.generate(
            idx, max_new_tokens=max_new_tokens, temperature=0.9, top_k=40, eos_id=tokenizer.eos_id
        )[0].tolist()
        response_ids = out_ids[idx.shape[1] :]
        hit_eos = tokenizer.eos_id in response_ids
        if hit_eos:
            response_ids = response_ids[: response_ids.index(tokenizer.eos_id)]
        completions.append({"text": tokenizer.decode(response_ids), "hit_eos": hit_eos})
    model.train()
    return completions


def group_advantages(rewards: torch.Tensor, eps: float = GROUP_ADV_EPS) -> torch.Tensor:
    """rewards: (batch_size, group_size). Normalizes each prompt's group of K
    rewards against that group's own mean/std — this is the GRPO baseline,
    standing in for a learned value function. If all K completions score the
    same (std=0), every advantage in that group is ~0: no gradient signal
    from a group with no separation, same failure mode dpo_data.py's pair
    builder had to explicitly skip for — here it just falls out naturally,
    no skip logic needed."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def append_generations(path: str, step: int, stage: str, rows: list[dict]) -> None:
    """Append eval/comparison samples to a local jsonl for blog visibility
    (W&B tables alone are awkward to cite). One file, `stage` discriminates
    pretrained/sft/dpo/grpo — reopen and filter rather than juggling paths."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {"step": step, "stage": stage, "prompt": row["prompt"], "output": row["output"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main(config_path: str):
    cfg: GRPOTrainConfig = load_grpo_train_config(config_path)
    assert cfg.model.type == "gpt" and cfg.model.tokenizer == "bpe", "train_grpo.py expects a bpe gpt model config"

    torch.manual_seed(cfg.seed)
    device = resolve_lm_device()
    OUT_DIR = os.path.join(BASE_DIR, cfg.output_dir or "out_grpo_bpe")
    os.makedirs(OUT_DIR, exist_ok=True)
    CHECKPOINTS_DIR = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    tokenizer = BPETokenizer()
    train_prompts = load_prompts(os.path.join(BASE_DIR, cfg.prompts_path))
    print(f"GRPO training prompts: {len(train_prompts)}")

    def build_model():
        return TinyGPT(
            vocab_size=tokenizer.vocab_size,
            block_size=cfg.model.block_size,
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            dropout=cfg.model.dropout,
        ).to(device)

    dpo_checkpoint = os.path.join(BASE_DIR, cfg.dpo_checkpoint)
    sft_checkpoint = os.path.join(BASE_DIR, cfg.sft_checkpoint)
    pretrained_checkpoint = os.path.join(BASE_DIR, cfg.pretrained_checkpoint)

    # Four-way qualitative comparison: base pretrained -> SFT -> DPO -> GRPO
    # (in progress), extending train_dpo.py's three-way comparison one stage
    # further, on the same fixed prompts throughout.
    pretrained_model = build_model()
    pretrained_model.load_state_dict(torch.load(pretrained_checkpoint, map_location=device))
    pretrained_samples = sample_prompts(pretrained_model, tokenizer, EVAL_PROMPTS, device, max_new_tokens=80)
    del pretrained_model

    sft_model_for_samples = build_model()
    sft_model_for_samples.load_state_dict(torch.load(sft_checkpoint, map_location=device))
    sft_samples = sample_prompts(sft_model_for_samples, tokenizer, EVAL_PROMPTS, device, max_new_tokens=80)
    del sft_model_for_samples

    dpo_model_for_samples = build_model()
    dpo_model_for_samples.load_state_dict(torch.load(dpo_checkpoint, map_location=device))
    dpo_samples = sample_prompts(dpo_model_for_samples, tokenizer, EVAL_PROMPTS, device, max_new_tokens=80)
    del dpo_model_for_samples

    # Reference — frozen for the entire run, same role DPO's reference
    # played: an anchor the KL term keeps the policy from drifting too far
    # from, not a target being optimized toward.
    reference = build_model()
    reference.load_state_dict(torch.load(sft_checkpoint, map_location=device))
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    # Policy — trainable; init from SFT so GRPO and DPO are siblings under
    # the same parent rather than stacking GRPO on DPO.
    policy = build_model()
    policy.load_state_dict(torch.load(sft_checkpoint, map_location=device))

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"training from: {sft_checkpoint}  (group_size={cfg.group_size}, kl_coef={cfg.kl_coef})")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

    run = init_run(
        {**dataclasses.asdict(cfg), "vocab_size": tokenizer.vocab_size, "n_params": n_params, "device": device},
        output_dir=OUT_DIR,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
    )
    # Explicit columns=["prompt", "output"]: log_samples' default (columns=None)
    # path builds a ["prompt", "base_output", "trained_output"] table meant for
    # scripts that log a base-vs-trained pair in one row. sample_prompts() only
    # ever returns {"prompt", "output"} — under the default path that leaves
    # "base_output" permanently blank in every table this script logs. Passing
    # columns explicitly keeps the W&B table honest: only real fields shown.
    SAMPLE_COLUMNS = ["prompt", "output"]
    log_samples(run, pretrained_samples, step=0, table_name="pretrained_samples", columns=SAMPLE_COLUMNS)
    log_samples(run, sft_samples, step=0, table_name="sft_samples", columns=SAMPLE_COLUMNS)
    log_samples(run, dpo_samples, step=0, table_name="dpo_samples", columns=SAMPLE_COLUMNS)

    generations_path = os.path.join(OUT_DIR, "generations.jsonl")
    # Fresh file each run (output_dir is overwritten by design for this recipe).
    open(generations_path, "w", encoding="utf-8").close()
    append_generations(generations_path, step=0, stage="pretrained", rows=pretrained_samples)
    append_generations(generations_path, step=0, stage="sft", rows=sft_samples)
    append_generations(generations_path, step=0, stage="dpo", rows=dpo_samples)

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(cfg.num_epochs):
        prompts = train_prompts.copy()
        for i in range(0, len(prompts), cfg.batch_size):
            batch_prompts = prompts[i : i + cfg.batch_size]
            if not batch_prompts:
                continue

            # --- rollout: sample a group of K completions per prompt from
            # the *current* policy. This is the on-policy step DPO never
            # needed — its pairs were fixed before training started.
            examples = []
            reward_rows = []  # (prompt_idx within this batch) -> list of rewards, same order as examples
            for p_idx, prompt in enumerate(batch_prompts):
                completions = sample_group(policy, tokenizer, prompt, cfg.group_size, device, cfg.max_new_tokens)
                rewards = [score_completion(c["text"], c["hit_eos"]) for c in completions]
                reward_rows.append(rewards)
                for c in completions:
                    examples.append(build_example(tokenizer, prompt, c["text"], cfg.model.block_size))

            rewards_t = torch.tensor(reward_rows, dtype=torch.float32, device=device)  # (batch, group_size)
            advantages = group_advantages(rewards_t).reshape(-1)  # (batch * group_size,) — flattened, same order as `examples`

            input_ids, labels, attention_mask = collate(tokenizer, examples, cfg.model.block_size, device)

            # mean (not sum): length-normalize so a 40-token reply doesn't get
            # ~4x the gradient of a 10-token one with the same per-token quality.
            policy_logprob = sequence_logprobs(policy, input_ids, labels, attention_mask, reduction="mean")
            with torch.no_grad():
                ref_logprob = sequence_logprobs(
                    reference, input_ids, labels, attention_mask, reduction="mean"
                )

            # Policy gradient step: push up the log-prob of completions this
            # group scored above its own average, push down the ones scored
            # below — group_advantages already centered this to zero-mean, so
            # "above/below" is relative to the K siblings, not an absolute bar.
            pg_loss = -(advantages.detach() * policy_logprob).mean()

            # KL-to-reference: simple sequence-level log-prob gap (see module
            # docstring for why not the exponentiated k3 estimator here).
            kl_penalty = (policy_logprob - ref_logprob).mean()

            loss = pg_loss + cfg.kl_coef * kl_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % cfg.eval_interval == 0:
                elapsed = time.time() - t0
                mean_reward = rewards_t.mean().item()
                reward_std = rewards_t.std(dim=1).mean().item()  # mean *within-group* std — signal separation health
                mean_len = sum(len(e["labels"]) for e in examples) / len(examples)
                eos_rate = sum(1 for row in reward_rows for r in row if r > 0) / (len(reward_rows) * cfg.group_size)
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "pg_loss": pg_loss.item(),
                        "kl_penalty": kl_penalty.item(),
                        "loss": loss.item(),
                        "mean_reward": mean_reward,
                        "reward_std": reward_std,
                        "mean_response_len": mean_len,
                        "elapsed_s": elapsed,
                    }
                )
                print(
                    f"epoch {epoch} step {step}: loss={loss.item():.4f} pg_loss={pg_loss.item():.4f} "
                    f"kl={kl_penalty.item():+.4f} mean_reward={mean_reward:+.3f} reward_std={reward_std:.3f} "
                    f"({elapsed:.0f}s)"
                )

                eval_samples = sample_prompts(policy, tokenizer, EVAL_PROMPTS, device, max_new_tokens=cfg.max_new_tokens)
                log_metrics(
                    run,
                    {
                        "train/pg_loss": pg_loss.item(),
                        "train/kl_penalty": kl_penalty.item(),
                        "train/loss": loss.item(),
                        "train/mean_reward": mean_reward,
                        "train/reward_std": reward_std,
                        "train/mean_response_len": mean_len,
                        "train/eos_rate": eos_rate,
                        "elapsed_s": elapsed,
                    },
                    step=step,
                )
                log_samples(run, eval_samples, step=step, table_name="grpo_samples", columns=SAMPLE_COLUMNS)

                # Print the actual text here too, not just the metrics above —
                # the numbers can look fine (loss dropping, reward climbing)
                # while the qualitative output is degrading (this is exactly
                # what happened during the DPO gold-dataset run: eval_acc
                # looked great while generation was catastrophically broken).
                # W&B tables are useful after the fact; this is what tells you
                # something's wrong *while the run is still going*.
                print(f"  --- sample outputs @ step {step} ---")
                for row in eval_samples:
                    print(f"  {row['prompt']!r} -> {row['output']!r}")
                append_generations(generations_path, step=step, stage="grpo", rows=eval_samples)

                torch.save(policy.state_dict(), os.path.join(CHECKPOINTS_DIR, f"step_{step}.pt"))

    torch.save(policy.state_dict(), os.path.join(OUT_DIR, "tinygpt_grpo.pt"))
    history_path = os.path.join(OUT_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump({"config": dataclasses.asdict(cfg), "history": history, "n_params": n_params}, f, indent=2)

    print(f"done. checkpoint -> {OUT_DIR}/tinygpt_grpo.pt")

    if history:
        plot_metric_groups(
            history,
            groups=[
                {
                    "metrics": ["pg_loss", "kl_penalty", "loss"],
                    "labels": {"pg_loss": "policy grad loss", "kl_penalty": "KL penalty", "loss": "total"},
                    "ylabel": "loss",
                    "title": "Loss",
                },
                {
                    "metrics": ["mean_reward"],
                    "labels": {"mean_reward": "mean reward"},
                    "ylabel": "reward (score_completion)",
                    "title": "Mean Reward",
                },
                {
                    "metrics": ["reward_std"],
                    "labels": {"reward_std": "within-group reward std"},
                    "ylabel": "std",
                    "title": "Group Reward Separation",
                },
            ],
            out_path=os.path.join(OUT_DIR, "training_curves.png"),
            title=f"GRPO on TinyGPT (bpe) — {n_params:,} params, group_size={cfg.group_size}",
        )

    final_samples = sample_prompts(policy, tokenizer, EVAL_PROMPTS, device, max_new_tokens=80)
    log_samples(run, final_samples, step=step, table_name="final_samples", columns=SAMPLE_COLUMNS)
    append_generations(generations_path, step=step, stage="grpo", rows=final_samples)
    finish_run(run)

    print("\n--- base pretrained vs SFT vs DPO vs GRPO, same prompts ---")
    for pre, sft, dpo, grpo in zip(pretrained_samples, sft_samples, dpo_samples, final_samples, strict=True):
        print(f"\nPrompt: {pre['prompt']}")
        print(f"  pretrained: {pre['output']!r}")
        print(f"  sft:        {sft['output']!r}")
        print(f"  dpo:        {dpo['output']!r}")
        print(f"  grpo:       {grpo['output']!r}")
    print(f"\ngenerations -> {generations_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/grpo_bpe.yaml")
    args = parser.parse_args()
    main(args.config)
