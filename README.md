# post-training-roadmap

Reusable repo scaffold for post-training experiments (SFT, DPO, GRPO).

The intent is to reuse the same config + logging conventions across all three projects.

## Stack (locked for 6 weeks)

- Python 3.11 + uv
- PyTorch, transformers, datasets, trl, peft, accelerate
- W&B for logging
- Primary model: `Qwen/Qwen2.5-0.5B-Instruct`
- Backup: `HuggingFaceTB/SmolLM2-360M-Instruct`

## Model Options (M4 24GB, fp16 + LoRA)

Memory budget rules of thumb:
- **SFT / GRPO training**: up to 3B comfortable, 7B tight (risky, avoid for training)
- **DPO training**: needs two model copies simultaneously — stay at 1.5B or below
- **Inference only**: up to 7B comfortably (~14GB fp16)

| Model | Size | fp16 RAM | Use |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~1GB | Fast iteration, debugging pipelines |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3GB | Sweet spot — good quality, fits DPO |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~6GB | Best quality for SFT/GRPO on this machine |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~2GB | Good DPO model (2 copies = ~4GB) |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~6GB | Stronger reasoning than Qwen 3B |
| `google/gemma-3-1b-it` | 1B | ~2GB | Recent, strong for its size |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14GB | **Inference only** — too tight for training |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | <1GB | Backup if 0.5B has issues |

> Note: `Qwen2.5-7B-Instruct` is listed in `configs/base.yaml` as a size option but is not safe for LoRA training at 24GB — use it for inference runs only.

**Suggested progression**: start with 0.5B to debug data pipelines (fast), switch to 1.5B or 3B once the loop is working.

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

### Dataset selection philosophy

> **Default principle: cheap and debuggable first.**
> Start with the smallest model that exposes the learning objective — `Qwen2.5-0.5B-Instruct` or `SmolLM2-360M/1.7B`. Use a narrow self-curated dataset or a small subset of an open dataset, not the full thing. Pick datasets where the training objective and the data signal have a clear, articulable connection — write down *why* each dataset matches the training objective before you start.

Concretely:
- **SFT**: start with a narrow self-curated dataset or a small subset of an open instruction dataset
- **DPO**: use a small pairwise preference set (≤5K pairs) and manually inspect 50 before training
- **RL/GRPO**: use a verifiable task dataset where the reward checker is trivial (exact number match, exact string match, valid JSON check)

### Recommended source datasets (by project)

**Project 1 — SFT (structured JSON output)**

| Dataset | HF ID | Notes |
|---|---|---|
| NER → JSON (recommended) | `conll2003` | Define output as `{"entities": [{"text": "...", "type": "PER"}]}`. Small (~14K train), verifiable, forces you to write the schema. |
| Function calling | `glaiveai/glaive-function-calling-v2` | Already structured as JSON, ~112K examples |
| Hermes function calling | `NousResearch/hermes-function-calling-v1` | Cleaner quality, ChatML format (matches Qwen template exactly) |

**Project 2 — DPO (preference learning)**

| Dataset | HF ID | Notes |
|---|---|---|
| UltraFeedback (recommended) | `HuggingFaceH4/ultrafeedback_binarized` | 64K chosen/rejected pairs, highest quality. Use a 5K subset to start. |
| Anthropic HH | `Anthropic/hh-rlhf` | Helpfulness + harmlessness pairs. Older but good baseline. |
| DPO Mix | `argilla/dpo-mix-7k` | 7K curated pairs, diverse topics. Good for smaller runs. |

**Project 3 — GRPO (RL with verifiable reward)**

| Dataset | HF ID | Notes |
|---|---|---|
| GSM8K (recommended) | `openai/gsm8k` | 7.5K math word problems, numeric answers. Verifier is ~5 lines: extract last number, compare. Exact match reward. |
| ARC-Easy | `allenai/ai2_arc` (easy split) | Multiple choice (A/B/C/D). Trivial verifier. Good fallback. |
| MATH | `lighteval/MATH` | Harder (AMC/AIME level). Use as stretch goal after GSM8K works. |

### Expected JSONL schemas (repo convention):
- `data/processed/sft_train.jsonl`, `data/eval/sft_eval.jsonl`
  - SFT train: `{"prompt": "...", "response": "..."}` (see `scripts/prepare_sft_data.py` for Project 1)
  - SFT eval: `{"prompt": "...", "ground_truth": "..."}` — canonical JSON string for exact-match eval (`eval_model.py`)
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
