# Project 1 Report — SFT: Structured Output

**Status:** baseline harness built; baseline sampled/scored
**Model:** Qwen/Qwen2.5-0.5B-Instruct
**Task:** CoNLL-style NER → strict JSON object
**Dataset:** `data/eval/sft_eval.jsonl`

---

## Metrics

| Metric | Base model | Trained |
|---|---|---|
| Format validity rate | 0.75 | |
| Exact field presence rate | 0.75 | |
| Task success rate | 0.00 | |
| Entity precision (micro) | 0.099 | |
| Entity recall (micro) | 0.184 | |
| Entity F1 (micro) | 0.129 | |
| Avg response length | 54.70 | |

## Eval config

Fixed generation params are read from `configs/base.yaml` for new eval/sample runs:
- `generation_max_new_tokens`: 256
- `generation_do_sample`: false
- `generation_temperature`: 0.0
- `num_sample_generations`: 20

Note: the current `reports/baseline_samples.jsonl` artifact was generated before this wiring used config-driven params, with `max_new_tokens=100`, `do_sample=false`, `temperature=0.0`. Its scored metrics are saved in `reports/baseline_metrics.json`.

Entity precision / recall / F1 are **micro** over all `(normalized text, type)` pairs in the first N eval rows: multiset overlap between predicted `entities` and gold (same definition as `eval_model.py`). Unparseable generations count as **no** predicted entities.

## Sample generations

- Baseline samples: `reports/baseline_samples.jsonl`
- Baseline metrics: `reports/baseline_metrics.json`
- Trained samples: `reports/project1_samples.jsonl` (pending)

## Failure analysis

Baseline failures are mostly schema/format failures:
- Many generations wrap JSON in markdown fences, which violates the “single JSON object only” instruction even when a JSON candidate is present.
- Several outputs use labels outside the allowed NER set (`VERB`, `DET`, `NOUN`, etc.).
- Some generations truncate before completing valid JSON, especially with the earlier `max_new_tokens=100` baseline run.

## Ablations
<!-- at least one ablation — e.g., LoRA rank 8 vs 16, epochs 1 vs 2 -->

## What I learned
<!-- honest writeup -->
