"""
Tests for bert_data.py's example construction — segment ids, special-token
handling, and that MLM masking never touches [CLS]/[SEP]/[PAD].
"""

from bert_data import CLS_ID, PAD_ID, SEP_ID, CharTokenizerWithSpecials, build_example


def _tokenizer():
    # Built from a tiny text sample — real char vocab, not a fake one, since
    # this module's special-token ids are baked in relative to the real
    # CharTokenizerWithSpecials construction.
    return CharTokenizerWithSpecials("ROMEO JULIET hello world")


def test_build_example_structure_is_cls_a_sep_b_sep():
    tok = _tokenizer()
    ex = build_example(tok, seg_a="hello", seg_b="world", nsp_label=0, block_size=32, mlm_probability=0.0)

    a_len = len(tok.encode("hello"))
    b_len = len(tok.encode("world"))

    assert ex["input_ids"][0] == CLS_ID
    assert ex["input_ids"][1 + a_len] == SEP_ID
    assert ex["input_ids"][1 + a_len + 1 + b_len] == SEP_ID


def test_build_example_segment_ids_split_correctly():
    tok = _tokenizer()
    ex = build_example(tok, seg_a="hello", seg_b="world", nsp_label=0, block_size=32, mlm_probability=0.0)

    a_len = len(tok.encode("hello"))
    b_len = len(tok.encode("world"))
    # [CLS] + seg_a + [SEP] all belong to segment 0; seg_b + [SEP] to segment 1.
    seg_a_region_len = 1 + a_len + 1
    seg_b_region_len = b_len + 1

    assert ex["segment_ids"][:seg_a_region_len] == [0] * seg_a_region_len
    assert ex["segment_ids"][seg_a_region_len : seg_a_region_len + seg_b_region_len] == [1] * seg_b_region_len


def test_build_example_padding_marked_in_attention_mask():
    tok = _tokenizer()
    ex = build_example(tok, seg_a="hello", seg_b="world", nsp_label=0, block_size=32, mlm_probability=0.0)

    real_len = sum(ex["attention_mask"])
    assert ex["attention_mask"][:real_len] == [1] * real_len
    assert ex["attention_mask"][real_len:] == [0] * (len(ex["attention_mask"]) - real_len)
    assert ex["input_ids"][real_len:] == [PAD_ID] * (len(ex["input_ids"]) - real_len)


def test_mlm_probability_zero_masks_nothing():
    tok = _tokenizer()
    ex = build_example(tok, seg_a="hello", seg_b="world", nsp_label=0, block_size=32, mlm_probability=0.0)
    assert all(label == -100 for label in ex["mlm_labels"])


def test_mlm_masking_never_touches_special_tokens():
    """Even at mlm_probability=1.0 (mask everything possible), CLS/SEP/PAD
    positions must never be selected for masking."""
    tok = _tokenizer()
    ex = build_example(tok, seg_a="hello", seg_b="world", nsp_label=0, block_size=32, mlm_probability=1.0)

    for i, tok_id in enumerate(ex["input_ids"]):
        if tok_id in (CLS_ID, SEP_ID, PAD_ID):
            assert ex["mlm_labels"][i] == -100, f"position {i} (special token {tok_id}) should never be a masking target"
