"""
Tests for sampling.py's sample_prompts_multi — added to disentangle genuine
policy degradation from single-draw sampling noise after the DPO
gold-dataset run's collapse turned out to need this distinction (see
reports/scratch_nanogpt.md).
"""

from bpe_data import BPETokenizer
from model import TinyGPT
from sampling import sample_prompts_multi


def _tiny_model(tokenizer):
    return TinyGPT(vocab_size=tokenizer.vocab_size, block_size=16, n_layer=1, n_head=1, n_embd=8, dropout=0.0)


def test_sample_prompts_multi_draws_k_rows_per_prompt():
    tokenizer = BPETokenizer()
    model = _tiny_model(tokenizer)
    prompts = ["ROMEO:", "JULIET:"]

    rows = sample_prompts_multi(model, tokenizer, prompts, device="cpu", k=3, max_new_tokens=5)

    assert len(rows) == len(prompts) * 3
    for prompt in prompts:
        draws = {row["draw"] for row in rows if row["prompt"] == prompt}
        assert draws == {0, 1, 2}


def test_sample_prompts_multi_leaves_model_in_train_mode():
    tokenizer = BPETokenizer()
    model = _tiny_model(tokenizer)
    model.train()

    sample_prompts_multi(model, tokenizer, ["ROMEO:"], device="cpu", k=2, max_new_tokens=5)

    assert model.training
