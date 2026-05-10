"""
Exact match and format validity metrics — shared across SFT/DPO/GRPO.

These are intentionally lightweight so they can run on CPU for evaluation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import json
import re
import statistics


def _extract_json_candidate(text: str) -> Optional[str]:
    """
    Attempt to extract the first JSON object/array substring from `text`.

    Heuristic: find the first '{' or '[' and the last matching '}' or ']'.
    """

    if not text:
        return None

    # Prefer objects if present.
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start == -1 and arr_start == -1:
        return None

    start = obj_start if (obj_start != -1 and (arr_start == -1 or obj_start < arr_start)) else arr_start
    if start == -1:
        return None

    # Last brace/bracket.
    obj_end = text.rfind("}")
    arr_end = text.rfind("]")
    end = obj_end if (obj_end != -1 and (arr_end == -1 or obj_end > arr_end)) else arr_end
    if end == -1 or end <= start:
        return None

    return text[start : end + 1]


def _try_parse_json(text: str) -> Optional[Any]:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


def is_parseable_json(text: str) -> bool:
    """True if `text` contains a JSON object/array substring that parses."""

    return _try_parse_json(text) is not None


def format_validity_rate(generations: List[str]) -> float:
    """Fraction of generations that contain a parseable JSON object/array."""

    if not generations:
        return 0.0
    valid = 0
    for g in generations:
        if is_parseable_json(g):
            valid += 1
    return valid / len(generations)


def exact_field_presence_rate(generations: List[str], required_fields: List[str]) -> float:
    """
    Fraction where *all* required_fields exist as keys in the parsed JSON object.

    If a generation parses to a non-object JSON (e.g. array/primitive), it counts as invalid.
    """

    if not generations:
        return 0.0
    if not required_fields:
        # No schema supplied; caller should skip this metric.
        return 0.0

    ok = 0
    for g in generations:
        parsed = _try_parse_json(g)
        if not isinstance(parsed, dict):
            continue
        if all(field in parsed for field in required_fields):
            ok += 1
    return ok / len(generations)


def task_success_rate(
    generations: List[str],
    ground_truths: List[str],
    verifier_fn: Callable[[str, str], float],
) -> float:
    """
    Average success score as returned by `verifier_fn(generation, ground_truth)`.

    If `ground_truths` is empty/mismatched length, returns 0.0.
    """

    if not ground_truths:
        return 0.0
    if len(generations) != len(ground_truths):
        # Avoid silently mis-scoring.
        raise ValueError("generations and ground_truths must have the same length")

    scores: List[float] = []
    for gen, gt in zip(generations, ground_truths):
        scores.append(float(verifier_fn(gen, gt)))
    return statistics.mean(scores) if scores else 0.0


def avg_response_length(generations: List[str]) -> float:
    if not generations:
        return 0.0
    lengths = [len(g.split()) for g in generations]
    return float(statistics.mean(lengths))

