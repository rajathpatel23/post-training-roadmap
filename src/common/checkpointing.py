"""
Shared checkpoint-resume helper, used by any training script in this repo
(pretraining, SFT, DPO, PPO/GRPO) that wants to optionally continue an
existing run's loss history instead of always restarting the plot at step 0.

Deliberately does NOT also decide "which weights to load" — that collapses
to a single `init_from` config field read directly by the caller (same
convention across every training script: init_from = the checkpoint that
training actually starts from / mutates). Continuing the *history* is an
independent decision from *which checkpoint init_from points at* — e.g.
continued pretraining onto a new corpus (train.py's Run 4) loads an
existing checkpoint but deliberately wants a fresh step-0 history, since
it's a new training stage, not a resumed interrupted run of the same stage.
Bundling both under one "resume_from is set" flag (an earlier version of
this module) doesn't generalize to that case — hence the explicit
`continue_history` flag instead of inferring it from any other config value.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def load_resumable_history(output_dir: str, continue_history: bool) -> Tuple[List[Dict[str, Any]], int]:
    """If continue_history, load the prior train_history.json from
    output_dir and return (history, last_step) so a new run continues the
    step counter and loss curve instead of restarting at step 0. Returns
    ([], 0) if continue_history is False, or if no history file exists yet."""
    if not continue_history:
        return [], 0
    history_path = os.path.join(output_dir, "train_history.json")
    if not os.path.exists(history_path):
        return [], 0
    with open(history_path) as f:
        history = json.load(f).get("history", [])
    start_step = history[-1]["step"] if history else 0
    return history, start_step
