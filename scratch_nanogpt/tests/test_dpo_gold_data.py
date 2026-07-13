"""
Tests for dpo_gold_data.py's pure-Python pieces: cost estimation, the
gold-cache/pairing logic, and the pending-task calculation shared by both
the sequential and batch generation paths. Does NOT call the real
Anthropic API — that part needs a live key and is exercised manually, not
in the test suite.
"""

import json

import dpo_gold_data as dgd


def test_estimate_cost_scales_linearly_with_prompt_count():
    cost_10 = dgd.estimate_cost(10, dgd.DEFAULT_MODEL)
    cost_100 = dgd.estimate_cost(100, dgd.DEFAULT_MODEL)
    assert cost_100 > cost_10
    assert cost_10 == 0 or abs(cost_100 / cost_10 - 10) < 1e-6


def test_estimate_cost_zero_calls_is_zero():
    assert dgd.estimate_cost(0, dgd.DEFAULT_MODEL) == 0.0


def test_load_cached_gold_returns_empty_dict_when_no_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dgd, "GOLD_CACHE_PATH", str(tmp_path / "does_not_exist.jsonl"))
    assert dgd.load_cached_gold() == {}


def test_load_cached_gold_groups_multiple_variants_per_prompt(tmp_path, monkeypatch):
    cache_path = tmp_path / "gold_completions.jsonl"
    cache_path.write_text(
        json.dumps({"prompt": "ROMEO:", "gold_chosen": "A fair line."}) + "\n"
        + json.dumps({"prompt": "JULIET:", "gold_chosen": "Another line."}) + "\n"
        + json.dumps({"prompt": "ROMEO:", "gold_chosen": "A second variant."}) + "\n"
    )
    monkeypatch.setattr(dgd, "GOLD_CACHE_PATH", str(cache_path))

    cached = dgd.load_cached_gold()
    assert cached == {
        "ROMEO:": ["A fair line.", "A second variant."],
        "JULIET:": ["Another line."],
    }


def test_pending_tasks_only_requests_missing_variants():
    cached = {"ROMEO:": ["variant 1", "variant 2"], "JULIET:": ["variant 1"]}
    tasks = dgd._pending_tasks(["ROMEO:", "JULIET:", "MENENIUS:"], cached, variants_per_prompt=2)

    # ROMEO: already has 2/2 -> no tasks. JULIET: has 1/2 -> needs 1 more.
    # MENENIUS: has 0/2 -> needs 2.
    assert tasks.count("ROMEO:") == 0
    assert tasks.count("JULIET:") == 1
    assert tasks.count("MENENIUS:") == 2


def test_pending_tasks_empty_when_every_prompt_already_at_target():
    cached = {"ROMEO:": ["v1", "v2", "v3"]}
    tasks = dgd._pending_tasks(["ROMEO:"], cached, variants_per_prompt=3)
    assert tasks == []


def _patch_sft_infra(monkeypatch, sample_fn):
    monkeypatch.setattr(dgd, "sample_k", sample_fn)
    monkeypatch.setattr(dgd, "resolve_lm_device", lambda: "cpu")

    class FakeTokenizer:
        vocab_size = 10

        def encode(self, s):
            return [1, 2, 3]

    monkeypatch.setattr(dgd, "BPETokenizer", FakeTokenizer)

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            pass

    monkeypatch.setattr(dgd, "TinyGPT", lambda **kwargs: FakeModel())
    monkeypatch.setattr(dgd.torch, "load", lambda *a, **k: {})


def test_build_pairs_skips_prompts_missing_gold_completions(monkeypatch):
    """A spend-cap-interrupted run leaves some prompts uncached — build_pairs
    must skip those instead of crashing on a missing key."""
    _patch_sft_infra(
        monkeypatch,
        lambda model, tokenizer, prompt, k, device: [{"text": f"sft sample for {prompt}", "hit_eos": True} for _ in range(k)],
    )

    prompts = ["ROMEO:", "JULIET:", "MENENIUS:"]
    gold = {"ROMEO:": ["A gold line for Romeo."], "JULIET:": ["A gold line for Juliet."]}
    # MENENIUS: has no gold completion (as if the spend cap stopped the run early).

    pairs = dgd.build_pairs(prompts, gold, k_rejected=2)

    prompts_in_pairs = {p["prompt"] for p in pairs}
    assert prompts_in_pairs == {"ROMEO:", "JULIET:"}
    assert "MENENIUS:" not in prompts_in_pairs
    assert len(pairs) == 4  # 2 prompts x 1 gold variant x 2 rejected samples each


def test_build_pairs_multiplies_across_gold_variants(monkeypatch):
    """Multiple gold variants for the same prompt each get their own set of
    paired rejected samples — that's the entire point of --variants-per-prompt."""
    _patch_sft_infra(
        monkeypatch,
        lambda model, tokenizer, prompt, k, device: [{"text": "sft sample.", "hit_eos": True} for _ in range(k)],
    )

    gold = {"ROMEO:": ["Variant one.", "Variant two.", "Variant three."]}
    pairs = dgd.build_pairs(["ROMEO:"], gold, k_rejected=2)
    assert len(pairs) == 6  # 3 gold variants x 2 rejected samples each
    assert {p["chosen"] for p in pairs} == {"Variant one.", "Variant two.", "Variant three."}


def test_build_pairs_skips_exact_ties_between_gold_and_sft_sample(monkeypatch):
    def fake_sample_k(model, tokenizer, prompt, k, device):
        # One sample happens to exactly match the gold completion (degenerate tie).
        return [{"text": "A gold line.", "hit_eos": True}, {"text": "A different sft sample.", "hit_eos": True}]

    _patch_sft_infra(monkeypatch, fake_sample_k)

    pairs = dgd.build_pairs(["ROMEO:"], {"ROMEO:": ["A gold line."]}, k_rejected=2)

    assert len(pairs) == 1  # the exact-tie sample was skipped
    assert pairs[0]["rejected"] == "A different sft sample."
