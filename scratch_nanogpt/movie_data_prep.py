"""
Build a flat text corpus from the Cornell Movie-Dialogs Corpus (ConvoKit
release), formatted to match tiny-shakespeare.txt's exact block structure
("CHARACTER:\\ndialogue\\n\\n") — same structural convention, new domain,
so continued pretraining reinforces a pattern the model already knows
instead of learning a new format and a new domain simultaneously.

Source: https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip
(the official ConvoKit-hosted release; 220K+ exchanges, 617 movies)

Schema (confirmed by inspection, not assumed):
- utterances.jsonl: {"id", "conversation_id", "text", "speaker", "reply-to", ...}
  reply-to points to the PREVIOUS utterance in the exchange (null = first).
- speakers.json: {speaker_id: {"meta": {"character_name": ..., "movie_name": ...}}}

Run:
    python movie_data_prep.py
"""

import json
import os
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "movie-corpus")
ZIP_PATH = os.path.join(DATA_DIR, "raw", "movie-corpus.zip")
ZIP_URL = "https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip"
OUT_PATH = os.path.join(DATA_DIR, "movie_dialogue.txt")


def download_and_extract():
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"downloading {ZIP_URL} ...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    if not os.path.exists(os.path.join(RAW_DIR, "utterances.jsonl")):
        print("extracting utterances.jsonl + speakers.json ...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extract("movie-corpus/utterances.jsonl", DATA_DIR + "/raw")
            zf.extract("movie-corpus/speakers.json", DATA_DIR + "/raw")


def load_speaker_names() -> dict[str, str]:
    with open(os.path.join(RAW_DIR, "speakers.json"), encoding="utf-8") as f:
        speakers = json.load(f)
    return {sid: info["meta"]["character_name"] for sid, info in speakers.items()}


def build_corpus_text() -> str:
    speaker_names = load_speaker_names()

    utterances = {}  # id -> {conversation_id, text, speaker}
    child_of = {}  # parent_id -> this utterance's id (reply-to chain, forward direction)
    roots_by_conv: dict[str, str] = {}  # conversation_id -> id of the first utterance

    with open(os.path.join(RAW_DIR, "utterances.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid, conv_id, text, speaker, reply_to = (
                row["id"],
                row["conversation_id"],
                row["text"].strip(),
                row["speaker"],
                row["reply-to"],
            )
            if not text:
                continue
            utterances[uid] = {"conversation_id": conv_id, "text": text, "speaker": speaker}
            if reply_to is None:
                roots_by_conv[conv_id] = uid
            else:
                child_of[reply_to] = uid

    print(f"loaded {len(utterances):,} utterances, {len(roots_by_conv):,} conversations")

    blocks = []
    for conv_id, root_id in roots_by_conv.items():
        uid = root_id
        seen = set()
        while uid is not None and uid not in seen:
            seen.add(uid)
            u = utterances.get(uid)
            if u is None:
                break
            name = speaker_names.get(u["speaker"], u["speaker"]).strip()
            if name:
                blocks.append(f"{name}:\n{u['text']}")
            uid = child_of.get(uid)

    return "\n\n".join(blocks) + "\n"


def main():
    download_and_extract()
    text = build_corpus_text()

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"wrote {len(text):,} chars -> {OUT_PATH}")
    print("\n--- first 500 chars, for a sanity check ---")
    print(text[:500])


if __name__ == "__main__":
    main()
