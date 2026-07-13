"""
Tests for sft_bpe_data.py's example construction — this is exactly where a
real bug lived (see reports/scratch_nanogpt.md's SFT Training section):
labels weren't shifted by one relative to input_ids, which trained the
model toward a degenerate "predict my own current token" objective and
produced repetition-collapse output. A unit test on build_example() would
have caught this instantly instead of discovering it via a training run.

Uses a small fake tokenizer (no network dependency, no real BPE vocab
needed) — these tests are about the shift/masking arithmetic, not about
what GPT-2's tokenizer actually produces.
"""

import torch

from sft_bpe_data import IGNORE_INDEX, build_example, collate


class FakeTokenizer:
    """One id per character (offset by 1 so real chars never collide with
    eos_id=0) — deterministic and network-free."""

    eos_id = 0
    vocab_size = 300

    def encode(self, s: str) -> list[int]:
        return [ord(c) + 1 for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i - 1) for i in ids if i != self.eos_id)


def test_build_example_shifts_labels_by_one():
    """The core bug: labels must predict the *next* token, not the current
    one — same convention as data.py's pretraining get_batch()."""
    tok = FakeTokenizer()
    prompt, response = "AB", "CD"
    ex = build_example(tok, prompt=prompt, response=response, block_size=100)

    full_ids = tok.encode(prompt) + tok.encode(response) + [tok.eos_id]
    expected_input_ids = full_ids[:-1]
    expected_targets = full_ids[1:]

    assert ex["input_ids"] == expected_input_ids
    assert len(ex["labels"]) == len(ex["input_ids"])
    for label, target in zip(ex["labels"], expected_targets, strict=True):
        if label != IGNORE_INDEX:
            assert label == target


def test_build_example_masks_prompt_region_only():
    """Loss must only apply to response tokens (+ EOS) — the model
    conditions on the prompt but isn't trained to reproduce it. The last
    prompt-token position IS a real target (it predicts the first response
    token, given the full prompt as context) — that's the off-by-one that
    made the original bug's fix non-trivial."""
    tok = FakeTokenizer()
    prompt, response = "AB", "CD"
    n_prompt = len(tok.encode(prompt))  # 2
    ex = build_example(tok, prompt=prompt, response=response, block_size=100)

    for i, label in enumerate(ex["labels"]):
        if (i + 1) < n_prompt:
            assert label == IGNORE_INDEX, f"position {i} should be masked (still predicting a prompt token)"
        else:
            assert label != IGNORE_INDEX, f"position {i} should be a real target"


def test_build_example_final_label_is_eos():
    """The model should learn to predict EOS right after the response —
    that's what actually gives generate() something to stop on."""
    tok = FakeTokenizer()
    ex = build_example(tok, prompt="A", response="B", block_size=100)
    assert ex["labels"][-1] == tok.eos_id


def test_build_example_respects_block_size_truncation():
    tok = FakeTokenizer()
    ex = build_example(tok, prompt="ABCDE", response="FGHIJ", block_size=4)
    assert len(ex["input_ids"]) <= 4
    assert len(ex["labels"]) == len(ex["input_ids"])


def test_collate_pads_labels_with_ignore_index_not_real_ids():
    """Padding must never leak into the loss — pad positions get
    IGNORE_INDEX regardless of what filler token sits in input_ids there."""
    tok = FakeTokenizer()
    examples = [
        build_example(tok, prompt="A", response="B", block_size=100),  # short
        build_example(tok, prompt="AB", response="CDEF", block_size=100),  # longer
    ]
    input_ids, labels, attention_mask = collate(tok, examples, block_size=100, device="cpu")

    assert input_ids.shape == labels.shape == attention_mask.shape
    # Wherever attention_mask is False (padding), the label must be IGNORE_INDEX.
    pad_positions = ~attention_mask
    assert torch.all(labels[pad_positions] == IGNORE_INDEX)
