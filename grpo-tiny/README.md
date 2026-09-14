# GRPO-Tiny: your first RL training run

A minimal, from-scratch implementation of **Group Relative Policy Optimization**
(GRPO, DeepSeek-R1) in PyTorch. A ~0.3M-parameter transformer you can read in an
afternoon learns to solve arithmetic word problems by trial, error, and
verifiable reward — no reward model, no human labels, no downloads.

## What each file does

| File | Concept |
|---|---|
| `config.py` | Every hyperparameter in one dataclass; CLI overrides (`--steps`, `--lr`, …) |
| `tokenizer.py` | Fixed 46-token word vocabulary: digits, punctuation, problem words. No training, no downloads |
| `data.py` | Word-problem generator (8 templates: add/subtract/multiply) + train/holdout splits via different seeds |
| `rewards.py` | The RLVR reward: 1.0 for the exactly-correct answer, +0.05/+0.05 for two format rungs ("answer" emitted, then "answer :" adjacent) |
| `model.py` | Decoder-only transformer from scratch (embedding → 2 pre-norm blocks → head) + `generate()` used for rollouts |
| `grpo.py` | The algorithm: `sample_group` (rollouts), `group_advantages` (group normalization), `grpo_update` (clipped surrogate + KL) |
| `train.py` | Training loop: sample → score → advantages → update → log. Writes `log.csv` + `model.pt` |
| `eval.py` | Greedy-decode the holdout set, report accuracy, show example generations |

## The GRPO math, in plain words

For **one prompt**, sample **G completions** from the current policy and score each
one (1 = right answer, 0 = wrong, +0.1 for using the answer format).

**Step 1 — advantages from the group, not a critic.** Subtract the group's mean
reward and divide by its standard deviation:

```
A_i = (r_i − mean(r)) / (std(r) + ε)
```

Completions better than their siblings get positive advantage, worse ones get
negative. If the whole group ties, every advantage is 0 and nothing is learned
from that prompt — correct behavior.

**Step 2 — move the policy, but not too far.** For each token, compute how much
more likely it is under the new policy than the old one:

```
ratio = π_new(token) / π_old(token)
```

and maximize `min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`. This is the PPO idea:
follow the advantage signal, but clip how far any single update can push a
token's probability. Average over tokens with a `1/|completion|` normalization
so long rambling answers don't dominate the gradient.

**Step 3 — optional KL leash.** With `--kl-beta > 0`, add a penalty on
`KL(π_new ‖ π_ref)` (k3 estimator) against a frozen reference copy, keeping the
policy from drifting into degenerate text.

That's the whole algorithm. Everything else is plumbing.

## Run it

Install (CPU-only torch is fine):

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Smoke test (~2–5 min on CPU — confirms the pipeline works):

```bash
cd ~/workspace/rl-sprint/grpo-tiny
python train.py --steps 12 --prompts-per-step 2 --group-size 4 --max-new-tokens 32
```

Full tiny run (~30–60 min on CPU):

```bash
python train.py --steps 200
```

Evaluate the checkpoint on held-out problems:

```bash
python eval.py --ckpt runs/tiny/model.pt
python eval.py --ckpt runs/tiny/model.pt --n 40   # quick check
python eval.py --ckpt runs/tiny/model.pt --samples 8 --n 60  # adds pass@8
```

Logs: `runs/tiny/log.csv` has one row per step
(`step, reward, kl, entropy, mean_len, loss, holdout_acc, secs`).

## What healthy curves look like

- **reward** starts near 0 (a random policy almost never lands the exact answer)
  and climbs as the format rungs are discovered: first +0.05 for emitting
  "answer", then the full "answer :" pattern, then correct answers push it
  toward 0.5–1.0. The format is learned *before* the math.
- **holdout_acc** should track the reward upward. If train reward rises but
  holdout accuracy stays flat, you're overfitting the training pool (widen it).
- **kl** (per-step `KL(π_new ‖ π_old)`) stays small (< 0.05). Spikes mean the
  update was too aggressive — lower `--lr` or `--eps-clip`.
- **entropy** starts near `ln(vocab)` (≈3.83 here) and decays slowly. If it
  collapses early, the policy stopped exploring (all completions identical →
  all advantages 0 → learning stalls). Raise `--temperature` or add an entropy
  bonus (exercise 3).
- **mean_len** often *grows* first (the model discovers that emitting the
  format pattern pays) then stabilizes. Runaway length with flat reward =
  reward hacking; shorten `max_new_tokens` or penalize length.

Red flags: reward flat at ~0.1 after a few hundred steps (only the format
rungs learned — the arithmetic signal may be too sparse; try more steps,
larger `--group-size`, or a slightly higher `--lr`); loss NaN (lower lr);
every group ties (raise temperature).

### Observed on this box (CPU, 2 cores)

A 150-step run (`--steps 150`, defaults) took ~4 minutes and showed the
textbook curriculum: reward 0.00 → ~0.07 as rung 1 ("answer") was learned
around step 60–70, first rung-2 ("answer :") hits near step 145, entropy
3.78 → 3.54, KL ≈ 0 throughout, holdout accuracy still 0.000 — format
learned, arithmetic not yet. An extended 300-step run
(`--steps 600 --group-size 8 --lr 3e-4`, stopped halfway for time) plateaued
at reward ~0.09–0.10 with the format fully mastered; correct answers began
appearing as rare sampling events (~1 per 40 steps from step ~115 on) but had
not yet consolidated into the greedy policy (holdout 0.000 at steps 99/199).
This staged learning — format first, then content — is exactly what RLVR
looks like in practice, including the honest part: sparse correctness signals
take many steps (or a pretrained instruct model) to fully consolidate.

### Two bugs caught during development (worth knowing)

1. **Advanced-indexing footgun.** `logp[P-1:P+L-1, comp_ids]` with a slice and
   a *list* does not gather one element per row — it returns an `(L, L)`
   matrix. The fix is two 1-D index tensors: `logp[rows, cols]` with
   `rows = torch.arange(P-1, P+L-1)`, `cols = torch.tensor(comp_ids)`.
   Caught because KL was ~0.03 on steps where the policy provably didn't move.
2. **Entropy must use the full distribution.** Averaging `-p·log p` over only
   the *sampled* tokens is not entropy; use
   `-(logp.exp() * logp).sum(-1).mean()` over the whole vocabulary.
   Caught because "entropy" read 0.08 for a near-uniform random policy
   (true value: ln(46) ≈ 3.83).

## Path to scale

This repo is a *learning* implementation. When the concepts are second nature:

1. **Same code, bigger tiny model.** Raise `d_model`/`n_layer` in `config.py`
   and harder problems in `data.py`. Still runs on CPU for a while.
2. **Swap in a real pretrained model on a cloud GPU.** Replace
   `TinyTransformer` with a HuggingFace model (e.g. `Qwen2.5-0.5B`), keep the
   `grpo.py` update logic, and run on a cheap GPU (Lambda/Vast.ai/Modal —
   an 0.5B model with LoRA fits on a single 24GB card). This is the single
   highest-leverage step: real tokenizer, real priors, real RL dynamics.
3. **Graduate to TRL, then verl.** HuggingFace TRL's `GRPOTrainer` is the
   natural next read — compare its implementation against this repo line by
   line; you'll understand every argument. When you want multi-GPU rollouts,
   vLLM generation, and production RLHF/RLVR at scale, move to `verl`
   (the open-source successor to OpenPPL) — that's what frontier labs' stacks
   rhyme with.
4. **Then the frontier patterns:** RLVR on verifiable domains (math, code),
   process rewards, RL on tool use / agentic rollouts, and evals that can't be
   gamed. This repo is step 0 of that ladder.

## Exercises (specs, not solutions — implement them yourself)

**Exercise 1 — KL penalty scheduling.**
`kl_beta` is currently fixed (default 0). Implement a schedule: start at 0 and
ramp linearly to 0.05 over the first half of training, then hold. You'll need
to (a) add `kl_beta_start`/`kl_beta_end`/`kl_ramp_steps` to `config.py`,
(b) compute the current beta from the step in `train.py` and pass it into
`grpo_update`, and (c) log the active beta in `log.csv`. Run twice (fixed 0 vs
scheduled) and compare: does the KL leash change final holdout accuracy, and
does it prevent the late-training entropy collapse? Write up 3 sentences.

**Exercise 2 — DPO on the same task.**
Build a preference dataset from your own rollouts: for each prompt, take one
completion with reward ≥ 1.0 (chosen) and one with reward < 0.1 (rejected);
discard prompts where no such pair exists. Write `dpo.py` implementing the DPO
loss from scratch (no TRL): `−log σ(β·[(log π_θ(y_w) − log π_ref(y_w)) −
(log π_θ(y_l) − log π_ref(y_l))])`, with π_ref a frozen copy of the initial
policy. Train with the same tiny model and compare the learning curve against
GRPO: which method is more sample-efficient here, and why do you think so?

**Exercise 3 — Entropy bonus for exploration.**
Add an entropy bonus to the GRPO objective: subtract `ent_beta * entropy` from
the loss (i.e. maximize entropy) with `ent_beta` in `config.py`. Sweep
`ent_beta ∈ {0, 0.001, 0.01}` for 100 steps each and plot reward vs entropy for
all three runs on the same axes. Identify the failure mode at each extreme
(collapse at 0? reward never takes off at 0.01?) and pick the value you'd
actually ship. Bonus: make the bonus *adaptive* — target a fixed entropy and
adjust `ent_beta` with a simple proportional controller.
