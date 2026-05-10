"""
Prompt templates for post-training experiments.

Project 1 (SFT): NER → fixed JSON schema (CoNLL-style labels).
"""

from __future__ import annotations

import json
from typing import Any

# CoNLL-2003 entity types after stripping B-/I- prefix.
_ENTITY_TYPES = frozenset({"PER", "ORG", "LOC", "MISC"})


def format_sft_ner_prompt(sentence: str) -> str:
    """
    User-facing instruction for structured NER output.

    Model must emit a single JSON object (no fences, no extra prose).
    """

    sentence = sentence.strip()
    return (
        "Extract named entities from the sentence below. "
        "Respond with a single JSON object only (no markdown, no explanation) "
        'using this exact schema:\n'
        '{"entities": [{"text": <string>, "type": <string>}]}\n'
        "Each type must be one of: PER, ORG, LOC, MISC. "
        "Use an empty list for entities if there are none.\n\n"
        f"Sentence: {sentence}"
    )


def entities_to_json_response(entities: list[dict[str, str]]) -> str:
    """Canonical JSON string for training labels and eval ground truth."""

    # Stable order for exact-match eval: by span text then type.
    sorted_entities = sorted(
        entities,
        key=lambda e: (e.get("text", ""), e.get("type", "")),
    )
    return json.dumps({"entities": sorted_entities}, ensure_ascii=False, sort_keys=True)


def conll_tokens_and_tags_to_entities(
    tokens: list[str],
    ner_tags: list[str],
) -> list[dict[str, str]]:
    """
    Merge BIO tags into {text, type} entities (types: PER, ORG, LOC, MISC).
    """

    entities: list[dict[str, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = ner_tags[i]
        if tag == "O" or not tag.startswith("B-"):
            i += 1
            continue
        ent_type = tag[2:]
        if ent_type not in _ENTITY_TYPES:
            i += 1
            continue
        start = i
        i += 1
        while i < n and ner_tags[i] == f"I-{ent_type}":
            i += 1
        span_tokens = tokens[start:i]
        text = " ".join(span_tokens)
        entities.append({"text": text, "type": ent_type})
    return entities


def format_pref_prompt(example: dict[str, Any]) -> str:
    """Placeholder for Project 2."""

    raise NotImplementedError("use prepare_pref_data + DPO pipeline")


def format_rl_prompt(example: dict[str, Any]) -> str:
    """Placeholder for Project 3."""

    raise NotImplementedError("use prepare_rl_data + GRPO pipeline")
