"""Verifiable rewards (the RLVR pattern).

No learned reward model: we extract the final answer from the completion and
compare it to the ground truth. Reward is:

    1.0   if the extracted answer is exactly correct
    0.0   otherwise
    +0.05 rung 1: the completion emits the word "answer" anywhere
    +0.05 rung 2: "answer" is immediately followed by ":" (the full format)

The two format rungs are *shaping* rewards: a from-scratch policy discovers
"answer" (~1 in 40 tokens) far sooner than the full "answer : <digits>"
pattern, so the rungs form a curriculum -- emit the word, then the format,
then the right number. This mirrors real RLVR pipelines, which always pair
the sparse correctness signal with a dense format reward.
"""

RUNG_1_BONUS = 0.05   # emits "answer"
RUNG_2_BONUS = 0.05   # "answer" immediately followed by ":"


def extract_answer(comp_ids, tok):
    """Find the LAST 'answer : [minus] <digits>' pattern in the completion.

    Returns the integer, or None if no well-formed answer is present.
    """
    ans, colon, minus = tok.stoi["answer"], tok.stoi[":"], tok.stoi["minus"]
    last = None
    for i in range(len(comp_ids) - 1):
        if comp_ids[i] == ans and comp_ids[i + 1] == colon:
            last = i
    if last is None:
        return None
    j = last + 2
    neg = False
    if j < len(comp_ids) and comp_ids[j] == minus:
        neg, j = True, j + 1
    ds = []
    while j < len(comp_ids) and tok.itos[comp_ids[j]].isdigit():
        ds.append(tok.itos[comp_ids[j]])
        j += 1
    if not ds:
        return None
    val = int("".join(ds))
    return -val if neg else val


def score_completion(comp_ids, tok, correct: int):
    """Returns (reward, info dict)."""
    ans, colon = tok.stoi["answer"], tok.stoi[":"]
    rung1 = ans in comp_ids
    rung2 = any(comp_ids[i] == ans and comp_ids[i + 1] == colon
                for i in range(len(comp_ids) - 1))
    pred = extract_answer(comp_ids, tok) if rung2 else None
    ok = (pred == correct) if pred is not None else False
    reward = (1.0 if ok else 0.0) + (RUNG_1_BONUS if rung1 else 0.0) \
        + (RUNG_2_BONUS if rung2 else 0.0)
    return reward, {"correct": ok, "formatted": rung2, "rung1": rung1}
