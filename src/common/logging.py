"""
W&B logging helpers.

We keep these lightweight so all three projects (SFT/DPO/GRPO) can share the same
logging conventions:
- init run + save local metadata (git SHA, config, timestamp)
- log metrics (dict)
- log sample generations (as a W&B Table)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import datetime as _dt
import json
import os
import subprocess
import uuid


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _try_get_git_commit(repo_dir: str | None = None) -> Optional[str]:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_dir is not None:
            out = subprocess.check_output(cmd, cwd=repo_dir, stderr=subprocess.DEVNULL)
        else:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        sha = out.decode("utf-8").strip()
        return sha or None
    except Exception:
        return None


def _ensure_dir(path: str | None) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def init_run(
    config: Dict[str, Any],
    *,
    output_dir: str | None = None,
    project: str | None = None,
    run_name: str | None = None,
    tags: List[str] | None = None,
    entity: str | None = None,
) -> Any | None:
    """
    Initialize a Weights & Biases run and write `run_metadata.json` locally.

    Returns the `wandb.Run` instance.
    """

    # Import lazily so unit tests / import-only use doesn't require wandb setup.
    import wandb

    # Resolve run naming defaults.
    project_name = project or os.environ.get("WANDB_PROJECT") or config.get("project") or "post-training-roadmap"
    run_name = run_name or config.get("run_name") or f"run-{uuid.uuid4().hex[:8]}"

    entity = entity or os.environ.get("WANDB_ENTITY") or None

    _ensure_dir(output_dir)
    # Serialize locally even if W&B credentials are missing; helps with reproducibility.
    metadata = {
        "run_id": uuid.uuid4().hex,
        "timestamp_utc": _utc_now_iso(),
        "project": project_name,
        "run_name": run_name,
        "git_commit": _try_get_git_commit(repo_dir=output_dir and os.path.dirname(output_dir)),
        "config": config,
    }

    # We set `dir` so artifacts (like wandb offline files) are written under `output_dir`.
    try:
        run = wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            tags=tags,
            config=config,
            dir=output_dir or None,
        )
    except Exception:
        # Keep the rest of the pipeline usable even when W&B isn't configured.
        return None

    if output_dir:
        meta_path = os.path.join(output_dir, "run_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return run


def log_metrics(run: Any, metrics: Dict[str, Any], *, step: int) -> None:
    if run is None:
        return
    # Keep a consistent naming convention: callers should provide `train/*`, `eval/*`, etc.
    run.log(metrics, step=step)


def log_samples(
    run: Any,
    samples: List[Dict[str, Any]],
    *,
    step: int,
    table_name: str,
) -> None:
    """
    Log sample generations to W&B as a table.

    Expected columns:
    - `prompt`
    - `base_output` (optional)
    - `trained_output` (optional)
    """

    if run is None:
        return

    import wandb

    columns = ["prompt", "base_output", "trained_output"]
    data_rows = []
    for s in samples:
        prompt = s.get("prompt")
        base_output = s.get("base_output", None)
        trained_output = s.get("trained_output", None)

        # Backwards-compatible fallback if only a single output exists.
        if base_output is None and "output" in s:
            trained_output = s.get("output")
        if trained_output is None and "output" in s:
            trained_output = s.get("output")

        data_rows.append([prompt, base_output, trained_output])

    table = wandb.Table(columns=columns, data=data_rows)
    run.log({table_name: table}, step=step)


def finish_run(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        # Best-effort; logging failures shouldn't crash training.
        pass


class WandbLossCallback:  # pragma: no cover (depends on transformers runtime)
    """
    Optional callback for HuggingFace Trainer-based loops.

    If you use TRL Trainer classes, they already support `report_to=wandb`, but this
    keeps the “train loss logged” requirement explicit.
    """

    def __init__(self, run: Any, *, metric_key: str = "train/loss"):
        self.run = run
        self.metric_key = metric_key

    # Transformers calls `on_log` with `logs` dict.
    def on_log(self, args: Any, state: Any, control: Any, logs: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        if "loss" not in logs:
            return
        step = int(getattr(state, "global_step", 0))
        try:
            self.run.log({self.metric_key: logs["loss"]}, step=step)
        except Exception:
            pass

