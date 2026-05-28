"""
Post-hoc checkpoint selection by entity F1 (Option C), then merge LoRA into base weights.

Training keeps `eval_loss` for validation curves; this script scores saved checkpoints
with greedy generation on the held-out eval set and picks the highest entity F1.

Run after training:
    python scripts/select_best_checkpoint.py \
        --config configs/sft/qwen05b_structured.yaml \
        --output_dir outputs/project1_sft \
        --eval_path data/eval/sft_eval.jsonl \
        --merged_dir outputs/project1_sft/checkpoint-best-f1-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.checkpoint_selection import select_best_by_entity_f1

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument(
        "--merged_dir",
        default=None,
        help="Where to write merged weights (default: <output_dir>/checkpoint-best-f1-merged)",
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="Only select best checkpoint; do not merge LoRA into base weights",
    )
    args = parser.parse_args()
    select_best_by_entity_f1(
        config_path=args.config,
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        merged_dir=args.merged_dir,
        merge=not args.no_merge,
        repo_root=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
