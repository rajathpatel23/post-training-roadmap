"""
Tests for src/common/checkpointing.py — specifically that continue_history
is independent of anything else (the altitude review's finding on the
original version of this module: it inferred history-continuation from
resume_from's truthiness, which didn't fit train.py's Run 4 case where a
checkpoint is loaded but history should still restart at step 0).
"""

import json

from src.common.checkpointing import load_resumable_history


def test_continue_history_false_always_returns_empty(tmp_path):
    history_path = tmp_path / "train_history.json"
    history_path.write_text(json.dumps({"history": [{"step": 500, "train_loss": 1.0}]}))

    history, start_step = load_resumable_history(str(tmp_path), continue_history=False)
    assert history == []
    assert start_step == 0


def test_continue_history_true_with_no_prior_file_returns_empty(tmp_path):
    history, start_step = load_resumable_history(str(tmp_path), continue_history=True)
    assert history == []
    assert start_step == 0


def test_continue_history_true_with_prior_file_resumes_from_last_step(tmp_path):
    prior_history = [{"step": 100, "train_loss": 2.0}, {"step": 200, "train_loss": 1.5}]
    (tmp_path / "train_history.json").write_text(json.dumps({"history": prior_history}))

    history, start_step = load_resumable_history(str(tmp_path), continue_history=True)
    assert history == prior_history
    assert start_step == 200
