"""Arithmetic word-problem generator with train/holdout splits.

The task: read a short word problem, think, and emit "answer : <digits>".
Because the answer is exactly checkable, rewards are *verifiable* -- this is
the RLVR (RL with Verifiable Rewards) setup behind modern reasoning models.

Every template only uses words from the tokenizer vocabulary.
"""
import random
from dataclasses import dataclass


@dataclass
class Problem:
    prompt: str   # text fed to the model, including the format instruction
    answer: int


# (template, answer_fn, max_operand). Numbers are rendered digit-by-digit so the
# tokenizer never sees an out-of-vocabulary number.
TEMPLATES = [
    ("what is {a} plus {b} ?", lambda a, b: a + b, 20),
    ("add {a} and {b} .", lambda a, b: a + b, 20),
    ("what is the sum of {a} and {b} ?", lambda a, b: a + b, 20),
    ("what is {a} minus {b} ?", lambda a, b: a - b, 20),
    ("subtract {b} from {a} .", lambda a, b: a - b, 20),
    ("what is the difference of {a} and {b} ?", lambda a, b: a - b, 20),
    ("what is {a} times {b} ?", lambda a, b: a * b, 12),
    ("multiply {a} by {b} .", lambda a, b: a * b, 12),
]

SUFFIX = " think step by step . then write answer :"


def digits(n: int) -> str:
    """Render an integer as space-separated digit tokens: 42 -> '4 2', -7 -> 'minus 7'."""
    s = "minus " if n < 0 else ""
    return s + " ".join(str(abs(n)))


def gen_problem(rng: random.Random) -> Problem:
    tmpl, fn, mx = rng.choice(TEMPLATES)
    a, b = rng.randint(0, mx), rng.randint(0, mx)
    prompt = tmpl.format(a=digits(a), b=digits(b)) + SUFFIX
    return Problem(prompt=prompt, answer=fn(a, b))


def make_pool(seed: int, n: int) -> list:
    """A reproducible pool of problems. Train and holdout use different seeds,
    so the model is always evaluated on problems it never trained on."""
    rng = random.Random(seed)
    return [gen_problem(rng) for _ in range(n)]
