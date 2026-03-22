"""
Dump N side-by-side generations (base vs. trained) for qualitative review.

YOUR JOB: implement.

Output format should be human-readable — either markdown or structured JSONL.
This gets saved to reports/ and referenced in your project report.

Suggested functions:
  dump_side_by_side(
      prompts: list[str],
      base_outputs: list[str],
      trained_outputs: list[str],
      output_path: str,
      n: int = 20
  ) -> None

  bucket_failures(
      generations: list[str],
      labels: list[str]   # manual or rule-based failure labels
  ) -> dict[str, int]     # failure_type -> count
"""
