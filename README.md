# post-training-roadmap

Reusable repo scaffold for post-training experiments (SFT, DPO, GRPO).

The intent is to reuse the same config + logging conventions across all three projects.

## Stack (locked for 6 weeks)

- Python 3.11 + uv
- PyTorch, transformers, datasets, trl, peft, accelerate
- W&B for logging
- Primary model: `Qwen/Qwen2.5-0.5B-Instruct`
- Backup: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Repo Layout (shared across projects)

Top-level:
- `configs/`: YAML experiment configs (`base.yaml` + project-specific overrides)
- `data/`: raw/processed/eval JSONL datasets
- `scripts/`: runnable entrypoints (train, eval, sample generations)
- `src/`: reusable code (configs/logging/metrics/verifier logic)
- `notebooks/`: ad-hoc analysis
- `evals/`: lightweight, repo-wide evaluation artifacts/docs
- `reports/`: final outputs referenced by project reports

Shared code:
- `src/common/config.py`: minimal YAML + dataclass config loader
- `src/common/io.py`: JSONL/YAML helpers + deep-merge
- `src/common/logging.py`: W&B init, run metadata, metrics + sample table logging
- `src/evals/`: metric implementations used by all projects

## Projects

| # | Type | Objective | Status |
|---|---|---|---|
| 1 | SFT | Structured output — train model to respond in fixed schema | not started |
| 2 | DPO | Preference learning — pairwise comparison dataset | not started |
| 3 | GRPO | RL with verifiable reward — exact checker | not started |

## Datasets

Expected JSONL schemas (repo convention):
- `data/processed/sft_train.jsonl`, `data/eval/sft_eval.jsonl`
  - SFT eval file currently contains `{"prompt": "..."}`
  - SFT train file is expected to contain `{"prompt": "...", "response": "..."}`
- `data/processed/pref_train.jsonl`, `data/eval/pref_eval.jsonl`
  - preference pairs: `{"prompt": <messages>, "chosen": <messages>, "rejected": <messages>}`
- `data/processed/rl_train.jsonl`, `data/eval/rl_eval.jsonl`
  - verifiable prompts: `{"prompt": <messages>, "ground_truth": "..."}` (used by the GRPO verifier)

Raw inputs go under `data/raw/` and are transformed into `data/processed/` by the corresponding prep scripts.

## Experiment Rules

1. Use configs for everything
   - no hardcoded hyperparams in scripts; read YAML via `src/common/config.py`.
2. Fixed evaluation conditions
   - evaluation generation params are stored in `configs/base.yaml` (greedy eval by default for reproducibility).
   - eval dataset ordering is preserved (no shuffle) and is never reshuffled after the initial split.
3. Determinism knobs
   - training seed comes from `config.training.seed` (base config default is `42`).
4. Logging + metadata (repro)
   - run metadata (config + git SHA + timestamp) is saved locally under the script output directory.
   - metrics and sample generations are logged to W&B using `src/common/logging.py`.

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
