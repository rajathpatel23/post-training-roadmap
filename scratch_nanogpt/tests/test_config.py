"""
Tests for config.py's typed schemas. The whole point of moving from plain
dicts to dataclasses was to catch typos/stale fields at load time instead of
letting a dict key silently mean the wrong thing deep inside a training run
(see reports/scratch_nanogpt.md's Code Review section — the `init_from`
dual-meaning issue). These tests lock that behavior in.
"""

import os

import pytest

from config import (
    BERTTrainConfig,
    DPOTrainConfig,
    GPTTrainConfig,
    SFTTrainConfig,
    load_bert_train_config,
    load_dpo_train_config,
    load_gpt_train_config,
    load_sft_train_config,
)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "train")


@pytest.mark.parametrize(
    "filename",
    ["gpt_small.yaml", "gpt_small_bpe.yaml", "gpt_bpe_movie_continued.yaml"],
)
def test_every_real_gpt_config_loads(filename):
    cfg = load_gpt_train_config(os.path.join(CONFIGS_DIR, filename))
    assert isinstance(cfg, GPTTrainConfig)
    assert cfg.model.type == "gpt"


def test_every_real_bert_config_loads():
    cfg = load_bert_train_config(os.path.join(CONFIGS_DIR, "bert_small.yaml"))
    assert isinstance(cfg, BERTTrainConfig)
    assert cfg.model.type == "bert"


@pytest.mark.parametrize("filename", ["sft_bpe.yaml", "sft_bpe_resume.yaml"])
def test_every_real_sft_config_loads(filename):
    cfg = load_sft_train_config(os.path.join(CONFIGS_DIR, filename))
    assert isinstance(cfg, SFTTrainConfig)
    assert cfg.model.type == "gpt"
    assert cfg.model.tokenizer == "bpe"


@pytest.mark.parametrize("filename", ["dpo_bpe.yaml", "dpo_bpe_gold.yaml", "dpo_bpe_gold_lowbeta.yaml"])
def test_dpo_config_loads(filename):
    cfg = load_dpo_train_config(os.path.join(CONFIGS_DIR, filename))
    assert isinstance(cfg, DPOTrainConfig)
    assert cfg.model.type == "gpt"
    assert cfg.model.tokenizer == "bpe"
    # sft_checkpoint (training) and pretrained_checkpoint (qualitative
    # reference only) must be genuinely different checkpoints, or the
    # three-way comparison collapses to two-way.
    assert cfg.sft_checkpoint != cfg.pretrained_checkpoint


def test_dpo_gold_config_points_at_the_gold_dataset():
    cfg = load_dpo_train_config(os.path.join(CONFIGS_DIR, "dpo_bpe_gold.yaml"))
    assert "dpo_gold" in cfg.dpo_train_path
    assert "dpo_gold" in cfg.dpo_eval_path
    assert cfg.output_dir != load_dpo_train_config(os.path.join(CONFIGS_DIR, "dpo_bpe.yaml")).output_dir


def test_sft_reference_checkpoint_defaults_to_none_when_unset():
    """sft_bpe.yaml (a fresh run) doesn't set reference_checkpoint — the
    caller (train_sft.py) is responsible for defaulting it to init_from.
    This just locks in that the raw config value is None, not silently
    something else."""
    cfg = load_sft_train_config(os.path.join(CONFIGS_DIR, "sft_bpe.yaml"))
    assert cfg.reference_checkpoint is None


def test_sft_resume_config_sets_reference_checkpoint_explicitly():
    """sft_bpe_resume.yaml continues training from an SFT checkpoint
    (init_from), so reference_checkpoint must be explicitly set to the
    original pretrained checkpoint — otherwise the base-vs-SFT qualitative
    comparison would silently become SFT-vs-SFT."""
    cfg = load_sft_train_config(os.path.join(CONFIGS_DIR, "sft_bpe_resume.yaml"))
    assert cfg.reference_checkpoint is not None
    assert cfg.reference_checkpoint != cfg.init_from
    assert cfg.continue_history is True


def test_unknown_field_raises_instead_of_silently_ignored(tmp_path):
    """A typo'd or stale config key must fail loudly at load time, not get
    silently dropped — this is the actual point of using dataclasses."""
    model_dir = tmp_path / "configs" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "tiny.yaml").write_text(
        "type: gpt\nblock_size: 8\nn_layer: 1\nn_head: 1\nn_embd: 8\ndropout: 0.0\n"
    )

    train_dir = tmp_path / "configs" / "train"
    train_dir.mkdir(parents=True)
    bad_config = train_dir / "bad.yaml"
    bad_config.write_text(
        "model_config: tiny\n"
        "batch_size: 1\nlearning_rate: 0.001\nmax_iters: 1\n"
        "eval_interval: 1\neval_iters: 1\nseed: 1\n"
        "wandb_project: p\nwandb_run_name: r\n"
        "this_field_does_not_exist: true\n"
    )

    import config as config_module

    original_configs_dir = config_module.CONFIGS_DIR
    config_module.CONFIGS_DIR = str(tmp_path / "configs")
    try:
        with pytest.raises(TypeError):
            load_gpt_train_config(str(bad_config))
    finally:
        config_module.CONFIGS_DIR = original_configs_dir
