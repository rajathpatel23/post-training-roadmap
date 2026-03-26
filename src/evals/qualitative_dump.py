"""
Qualitative dump helpers (for side-by-side generation review).
"""

from __future__ import annotations

from typing import Dict, List
import json
import os


def bucket_failures(generations: List[str], labels: List[str]) -> Dict[str, int]:
    if len(generations) != len(labels):
        raise ValueError("generations and labels must have the same length")
    out: Dict[str, int] = {}
    for lab in labels:
        out[lab] = out.get(lab, 0) + 1
    return out


def dump_side_by_side(
    *,
    prompts: List[str],
    base_outputs: List[str],
    trained_outputs: List[str],
    output_path: str,
    n: int = 20,
) -> None:
    if not (len(prompts) == len(base_outputs) == len(trained_outputs)):
        raise ValueError("prompts, base_outputs, and trained_outputs must have the same length")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(min(n, len(prompts))):
            rec = {
                "idx": i,
                "prompt": prompts[i],
                "base_output": base_outputs[i],
                "trained_output": trained_outputs[i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

