## Evals (repo-wide)

This repo keeps *shared* evaluation utilities and scripts in two places:

1. `src/evals/`: reusable metric implementations (format validity, exact match, dumps).
2. `scripts/eval_model.py`: an end-to-end “generate + score + log” entrypoint for comparing a checkpoint vs the base model.

Why the extra top-level `evals/` folder?
- It makes it easy to add lightweight evaluation artifacts later (for example, extra eval configs, ablation configs, or notebooks) without mixing them into `src/`.

