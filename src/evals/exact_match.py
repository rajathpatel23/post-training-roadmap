"""
Exact match and format validity metrics — shared across SFT/DPO/GRPO.

These are intentionally lightweight so they can run on CPU for evaluation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
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


def json_structural_match(generation: str, ground_truth: str) -> float:
    """
    Return 1.0 when generation and ground_truth parse to the same JSON value.

    This ignores harmless formatting differences in otherwise equivalent JSON.
    """

    parsed_generation = _try_parse_json(generation)
    parsed_ground_truth = _try_parse_json(ground_truth)
    if parsed_generation is None or parsed_ground_truth is None:
        return 0.0
    return 1.0 if parsed_generation == parsed_ground_truth else 0.0


def json_structural_match_rate(generations: List[str], ground_truths: List[str]) -> float:
    """Fraction of generations whose parsed JSON exactly matches ground truth JSON."""

    return task_success_rate(generations, ground_truths, json_structural_match)


def _normalize_entity_text(text: str) -> str:
    """Collapse whitespace for stable span matching."""

    return " ".join(str(text).split())


def _entity_pairs_from_parsed(obj: Any) -> List[Tuple[str, str]]:
    """
    Extract (text, type) pairs from a parsed JSON object with an `entities` list.

    Skips malformed entries. Types are kept as stripped strings (case-sensitive).
    """

    if not isinstance(obj, dict):
        return []
    raw = obj.get("entities")
    if not isinstance(raw, list):
        return []
    out: List[Tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        typ = item.get("type")
        if text is None or typ is None:
            continue
        out.append((_normalize_entity_text(str(text)), str(typ).strip()))
    return out


def _entity_counter_from_json_text(text: str) -> Optional[Counter]:
    """
    Multiset of (normalized_text, type) from a model string or JSON ground truth.

    Returns None only if `text` is unparsable as JSON (no object/array substring).
    If JSON parses but has no usable `entities`, returns an empty Counter.
    """

    parsed = _try_parse_json(text)
    if parsed is None:
        return None
    return Counter(_entity_pairs_from_parsed(parsed))


def entity_tp_fp_fn(pred: Counter, gold: Counter) -> Tuple[int, int, int]:
    """One-example TP / FP / FN for multiset entity overlap."""

    keys = set(pred) | set(gold)
    tp = sum(min(pred[k], gold[k]) for k in keys)
    fp = pred.total() - tp
    fn = gold.total() - tp
    return tp, fp, fn


def entity_micro_prf1(generations: List[str], ground_truths: List[str]) -> Tuple[float, float, float]:
    """
    Micro-averaged precision, recall, and F1 over (text, type) entity pairs.

    - Pairs use normalized text (whitespace collapsed) and exact type string.
    - Unparseable generations contribute no predicted entities (all gold entities are FN).
    - Empty gold and empty pred: precision = recall = F1 = 1.0.
    """

    if len(generations) != len(ground_truths):
        raise ValueError("generations and ground_truths must have the same length")

    total_tp = total_fp = total_fn = 0
    for gen, gt in zip(generations, ground_truths):
        gold_c = _entity_counter_from_json_text(gt)
        if gold_c is None:
            continue
        pred_c = _entity_counter_from_json_text(gen)
        if pred_c is None:
            pred_c = Counter()
        tp, fp, fn = entity_tp_fp_fn(pred_c, gold_c)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    pred_support = total_tp + total_fp
    gold_support = total_tp + total_fn

    if pred_support == 0 and gold_support == 0:
        return 1.0, 1.0, 1.0
    precision = total_tp / pred_support if pred_support else (0.0 if gold_support > 0 else 1.0)
    recall = total_tp / gold_support if gold_support else (0.0 if pred_support > 0 else 1.0)
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def avg_response_length(generations: List[str]) -> float:
    if not generations:
        return 0.0
    lengths = [len(g.split()) for g in generations]
    return float(statistics.mean(lengths))

