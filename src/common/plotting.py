"""
Generic training-curve plotting, shared across every experiment (SFT/DPO/GRPO
reports, scratch_nanogpt, future projects). One shared style so plots look
consistent regardless of which project produced them — callers just describe
which metrics go in which subplot; this module doesn't know what a "loss" or
an "accuracy" is.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

PALETTE = ["#3b82f6", "#f97316", "#10b981", "#a855f7"]  # consistent color order across all plots


def plot_metric_groups(
    history: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    out_path: str,
    title: str = "",
    step_key: str = "step",
) -> None:
    """
    history: list of dicts, each with `step_key` + arbitrary metric keys
        (e.g. [{"step": 0, "train_loss": 2.1, "val_loss": 2.3}, ...])
    groups: one dict per subplot —
        {
          "metrics": ["train_loss", "val_loss"],   # required
          "labels": {"train_loss": "train"},       # optional, else metric name is used
          "ylabel": "loss", "title": "Loss",        # optional
          "hline": 0.5, "hline_label": "chance",    # optional reference line
          "ylim": (0, 1),                           # optional
        }
    """
    steps = [h[step_key] for h in history]
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        labels = group.get("labels", {})
        for i, metric in enumerate(group["metrics"]):
            values = [h[metric] for h in history]
            ax.plot(steps, values, label=labels.get(metric, metric), color=PALETTE[i % len(PALETTE)])
        if "hline" in group:
            ax.axhline(group["hline"], color="gray", linestyle="--", linewidth=1, label=group.get("hline_label", ""))
        if "ylim" in group:
            ax.set_ylim(*group["ylim"])
        ax.set_xlabel("step")
        ax.set_ylabel(group.get("ylabel", ""))
        ax.set_title(group.get("title", ""))
        ax.legend()
        ax.grid(alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


def exp_transform(history: List[Dict[str, Any]], source_key: str, target_key: str, cap: float = 20.0) -> None:
    """In-place: adds `target_key = exp(min(history[i][source_key], cap))` to
    every entry, so perplexity (or similar exp-of-loss metrics) can be plotted
    with the same plot_metric_groups() call as everything else."""
    import math

    for h in history:
        h[target_key] = math.exp(min(h[source_key], cap))
