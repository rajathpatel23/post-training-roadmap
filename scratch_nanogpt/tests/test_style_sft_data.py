"""The style pair keeps the movie line in the prompt and leaves the rewrite
for the response. These tests cover that split, not the API call."""

import os

from style_sft_data import eligible, iter_utterances, make_prompt, select_lines


def test_prompt_holds_the_movie_line_and_marks_the_rewrite():
    prompt = make_prompt("BIANCA", "They do not!")
    assert "BIANCA: They do not!" in prompt
    assert prompt.endswith("PLAY:")
    assert "MOVIE:" in prompt


def test_eligible_rejects_lines_too_short_to_carry_meaning():
    assert eligible("No") is False
    assert eligible("I hope you know what you did to me.") is True


def test_iter_utterances_reads_speaker_blocks(tmp_path):
    path = tmp_path / "movie.txt"
    path.write_text("BIANCA:\nThey do not!\n\nCAMERON:\nShe okay?\n", encoding="utf-8")
    assert list(iter_utterances(str(path))) == [
        ("BIANCA", "They do not!"),
        ("CAMERON", "She okay?"),
    ]


def test_train_and_eval_do_not_share_a_line():
    movie = os.path.join(os.path.dirname(__file__), "..", "data", "movie_dialogue.txt")
    if not os.path.exists(movie):
        return
    train_rows, eval_rows = select_lines(movie, n_train=20, n_eval=10, seed=1337)
    train_text = {row["text"] for row in train_rows}
    eval_text = {row["text"] for row in eval_rows}
    assert train_text.isdisjoint(eval_text)
    assert len(train_rows) == 20
    assert len(eval_rows) == 10
