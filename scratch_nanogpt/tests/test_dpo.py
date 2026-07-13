"""
Tests for the DPO-specific pieces: dpo_data.py's rule-based scoring
heuristic, and train_dpo.py's loss/reward math.
"""

import torch

from dpo_data import score_completion
from train_dpo import dpo_loss, sequence_logprobs
from sft_bpe_data import IGNORE_INDEX


def test_score_prefers_eos_over_rambling():
    assert score_completion("A complete sentence.", hit_eos=True) > score_completion(
        "A complete sentence.", hit_eos=False
    )


def test_score_prefers_terminal_punctuation():
    with_punct = score_completion("A complete sentence.", hit_eos=True)
    without_punct = score_completion("A complete sentence", hit_eos=True)
    assert with_punct > without_punct


def test_score_penalizes_repeated_trigrams():
    repeating = "the man the man the man went home today finally."
    clean = "The man went home today after a long journey finally."
    assert score_completion(clean, hit_eos=True) > score_completion(repeating, hit_eos=True)


def test_score_penalizes_degenerate_near_empty_text():
    assert score_completion("A real sentence here.", hit_eos=True) > score_completion(".", hit_eos=True)


def test_dpo_loss_is_ln2_when_policy_equals_reference():
    """If the policy hasn't diverged from the reference at all, the
    preference margin term is exactly 0, so loss = -log(sigmoid(0)) = ln(2)
    — a known, checkable reference point for the loss formula itself."""
    policy_chosen = torch.tensor([-5.0, -3.0])
    policy_rejected = torch.tensor([-6.0, -4.0])
    ref_chosen = policy_chosen.clone()
    ref_rejected = policy_rejected.clone()

    loss, chosen_reward, rejected_reward, accuracy = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )
    assert torch.isclose(loss, torch.tensor(0.6931), atol=1e-3)
    assert torch.isclose(chosen_reward, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(rejected_reward, torch.tensor(0.0), atol=1e-6)


def test_dpo_loss_decreases_as_policy_prefers_chosen_more():
    ref_chosen = torch.tensor([-5.0])
    ref_rejected = torch.tensor([-5.0])

    # Policy barely prefers chosen over rejected relative to reference.
    loss_small_margin, *_ = dpo_loss(
        torch.tensor([-4.9]), torch.tensor([-5.0]), ref_chosen, ref_rejected, beta=0.1
    )
    # Policy strongly prefers chosen over rejected relative to reference.
    loss_large_margin, *_ = dpo_loss(
        torch.tensor([-3.0]), torch.tensor([-6.0]), ref_chosen, ref_rejected, beta=0.1
    )
    assert loss_large_margin < loss_small_margin


def test_accuracy_reflects_whether_policy_reward_favors_chosen():
    ref_chosen = torch.tensor([-5.0, -5.0])
    ref_rejected = torch.tensor([-5.0, -5.0])
    # First pair: policy raises chosen's prob (correct preference).
    # Second pair: policy raises rejected's prob instead (wrong preference).
    policy_chosen = torch.tensor([-4.0, -5.0])
    policy_rejected = torch.tensor([-5.0, -4.0])

    _, _, _, accuracy = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert torch.isclose(accuracy, torch.tensor(0.5))


def test_sequence_logprobs_ignores_masked_positions():
    """A tiny hand-built 'model' (just returns fixed logits) lets us verify
    the masking arithmetic in isolation from any real network."""

    class FixedLogitsModel:
        def __call__(self, input_ids, key_padding_mask=None):
            batch, seq_len = input_ids.shape
            vocab_size = 5
            logits = torch.zeros(batch, seq_len, vocab_size)
            logits[:, :, 1] = 10.0  # token id 1 gets all the probability mass
            return logits, None

    model = FixedLogitsModel()
    input_ids = torch.tensor([[0, 0, 0]])
    attention_mask = torch.tensor([[True, True, True]])

    # Only position 2 is a real target (token id 1, which the model assigns
    # near-certain probability to) — positions 0, 1 are masked (prompt).
    labels_one_real_target = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1]])
    logprob_one_target = sequence_logprobs(model, input_ids, labels_one_real_target, attention_mask)

    # Same setup, but position 0 is also a real target for a token (id 0)
    # the model assigns near-zero probability to — logprob must be much lower.
    labels_two_targets = torch.tensor([[0, IGNORE_INDEX, 1]])
    logprob_two_targets = sequence_logprobs(model, input_ids, labels_two_targets, attention_mask)

    assert logprob_two_targets.item() < logprob_one_target.item()
