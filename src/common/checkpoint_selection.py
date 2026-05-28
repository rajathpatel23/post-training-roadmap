"""
Post-hoc checkpoint selection by entity F1 (Option C), with optional LoRA merge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.config import ExperimentConfig
from src.common.generation import resolve_lm_device
from src.common.io import read_jsonl
from src.common.model_loading import (
    is_peft_adapter_path,
    load_causal_lm_for_inference,
    load_tokenizer,
    merge_lora_and_save,
    release_model,
)
from src.evals.generative_eval import generate_completions, score_generations

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _abs_path(p: str, repo_root: Path | None = None) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    root = repo_root or Path.cwd()
    return root / path


def _checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_RE.match(path.name)
    if match:
        return int(match.group(1))
    return -1


def discover_checkpoints(output_dir: Path) -> list[Path]:
    """
    List checkpoint dirs to score, oldest step first.

    Includes `checkpoint-*` subdirs and the run root when it holds an adapter.
    """

    candidates: list[Path] = []
    if output_dir.is_dir():
        for child in sorted(output_dir.iterdir()):
            if child.is_dir() and _CHECKPOINT_RE.match(child.name):
                if is_peft_adapter_path(child) or (child / "config.json").is_file():
                    candidates.append(child)
        if is_peft_adapter_path(output_dir):
            candidates.append(output_dir)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in sorted(candidates, key=lambda x: (_checkpoint_step(x), x.name)):
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(p)
    return ordered


def select_best_by_entity_f1(
    *,
    config_path: str,
    output_dir: str,
    eval_path: str,
    merged_dir: str | None = None,
    merge: bool = True,
    repo_root: Path | None = None,
) -> dict:
    cfg = ExperimentConfig.from_yaml(config_path)
    out = _abs_path(output_dir, repo_root)
    eval_abs = _abs_path(eval_path, repo_root)
    records = read_jsonl(str(eval_abs))
    prompts = [r["prompt"] for r in records if "prompt" in r]
    ground_truths = [str(r["ground_truth"]) for r in records if "ground_truth" in r]
    if not prompts:
        raise ValueError(f"No prompts in {eval_path}")
    if len(ground_truths) != len(prompts):
        raise ValueError("Eval rows must include ground_truth for entity F1 selection")

    checkpoints = discover_checkpoints(out)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {out}")

    device = resolve_lm_device()
    fp16 = cfg.training.fp16
    bf16 = cfg.training.bf16
    required_fields = cfg.eval.required_fields or []

    scored: list[dict] = []
    for ckpt in checkpoints:
        ckpt_str = str(ckpt.resolve())
        tokenizer = load_tokenizer(ckpt_str, base_model=cfg.model.primary)
        model = load_causal_lm_for_inference(
            ckpt_str,
            base_model=cfg.model.primary,
            device=device,
            fp16=fp16,
            bf16=bf16,
        )
        outputs = generate_completions(
            model,
            tokenizer,
            prompts,
            device=device,
            max_new_tokens=cfg.eval.generation_max_new_tokens,
            do_sample=cfg.eval.generation_do_sample,
            temperature=cfg.eval.generation_temperature,
        )
        metrics = score_generations(
            outputs,
            ground_truths,
            required_fields=required_fields,
        )
        release_model(model)
        scored.append(
            {
                "checkpoint": ckpt.name,
                "checkpoint_path": ckpt_str,
                "metrics": metrics,
            }
        )
        print(
            f"{ckpt.name}: entity_f1={metrics['entity_f1']:.4f} "
            f"format_validity={metrics['format_validity_rate']:.4f}"
        )

    best = max(scored, key=lambda row: row["metrics"]["entity_f1"])
    best_path = best["checkpoint_path"]
    print(
        f"\nBest by entity_f1: {best['checkpoint']} "
        f"(F1={best['metrics']['entity_f1']:.4f})"
    )

    merged_path: str | None = None
    if merge:
        if merged_dir is None:
            merged_dir = str(out / "checkpoint-best-f1-merged")
        merged_abs = _abs_path(merged_dir, repo_root)
        if is_peft_adapter_path(best_path):
            merged_path = merge_lora_and_save(
                best_path,
                base_model=cfg.model.primary,
                output_dir=str(merged_abs),
                fp16=fp16,
                bf16=bf16,
            )
            print(f"Merged checkpoint written to {merged_path}")
        else:
            print(f"Skipping merge — {best_path} is not a PEFT adapter")

    report = {
        "selection_metric": "entity_f1",
        "strategy": "post_hoc_generative_eval",
        "config": config_path,
        "output_dir": str(out),
        "eval_path": str(eval_abs),
        "base_model": cfg.model.primary,
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_path": best_path,
        "best_metrics": best["metrics"],
        "merged_checkpoint_path": merged_path,
        "all_checkpoints": scored,
    }
    report_path = out / "checkpoint_selection.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Selection report: {report_path}")
    return report
