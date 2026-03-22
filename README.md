# post-training-roadmap

A personal 6-week program to build hands-on post-training skills across SFT, DPO, and GRPO.

## Stack (locked for 6 weeks)

- Python 3.11 + uv
- PyTorch, transformers, datasets, trl, peft, accelerate
- W&B for logging
- Primary model: `Qwen/Qwen2.5-0.5B-Instruct`
- Backup: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Projects

| # | Type | Objective | Status |
|---|---|---|---|
| 1 | SFT | Structured output — train model to respond in fixed schema | not started |
| 2 | DPO | Preference learning — pairwise comparison dataset | not started |
| 3 | GRPO | RL with verifiable reward — exact checker | not started |

## Week 1 deliverables

- [ ] Day 1: Repo, env, config structure, W&B project, README
- [ ] Day 2: Base model inference on 20 prompts, save to file
- [ ] Day 3: Choose Project 1 task, define dataset schema
- [ ] Day 4: Prepare train/eval split, write eval script stub
- [ ] Day 5: Baseline failure analysis, 1-page memo in `reports/`

## Setup

```bash
bash scripts/setup_env.sh
cp .env.example .env   # fill in WANDB credentials
source .venv/bin/activate
```

## Success criteria

For each project:
1. Base-model vs. trained comparison on a fixed eval set
2. One report with metrics, samples, and failure analysis
3. Explanation of why the objective matched the signal
4. At least one ablation

## Notes

See `notes/HOW_TO_USE_NOTES.md` — all learning notes go there, not here.
