"""
Tokenization helpers.

YOUR JOB: implement.

Things to understand before writing this:
  - How does Qwen2.5-Instruct's chat template work?
  - What is the difference between tokenizing the full sequence vs. masking the prompt tokens?
    (For SFT, you only want loss on the response tokens — understand how trl handles this)
  - What is the effect of truncation on your training examples?

Suggested functions:
  load_tokenizer(model_name: str) -> AutoTokenizer
  tokenize_sft_example(example: dict, tokenizer, max_length: int) -> dict
"""
