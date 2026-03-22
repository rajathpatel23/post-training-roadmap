"""
Verifier for Project 3 — exact-match reward signal.

YOUR JOB: implement.

The verifier is the reward function for GRPO. It takes a model generation
and ground truth, and returns a scalar reward.

Key design decisions to document in notes/project3_design.md:
  - What counts as a valid generation? (format check first, then correctness)
  - How do you handle partial credit vs. binary reward?
  - What reward value for invalid format vs. wrong answer vs. correct answer?
    (e.g., -1 / 0 / +1 or 0 / 0.5 / 1.0 — understand why this matters for GRPO)

Suggested functions:
  verify(generation: str, ground_truth: str) -> float
    # returns reward scalar
  is_valid_format(generation: str) -> bool
    # check format before checking correctness
  extract_answer(generation: str) -> str | None
    # pull the answer field from structured output
"""
