"""
File I/O utilities used across projects.

This repo is shared across SFT/DPO/GRPO, so keep these helpers small and consistent.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List
import json
import os
import yaml


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(obj)}")
    return obj


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def merge_configs(base_path: str, override_path: str) -> Dict[str, Any]:
    """
    Deep-merge `override_path` into `base_path`.

    Note: this is a generic helper; if your project YAML uses `defaults`, prefer
    `src.common.config.ExperimentConfig.from_yaml`.
    """

    base_cfg = load_yaml(base_path)
    override_cfg = load_yaml(override_path)
    # Avoid mutating inputs; keep deterministic merge semantics.
    return _deep_merge_dicts(base_cfg, override_cfg)

