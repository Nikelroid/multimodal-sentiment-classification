"""Shared access to preprocessed MELD manifests (see preprocess_meld.py).

Adds dialogue context to each utterance (previous N utterances of the same
dialogue, joined with the separator token) and resolves audio / frame / face
paths. Sentiment uses the project encoding (neutral: 0, negative: 1,
positive: 2); emotion ids follow the FER2013 label order so the face model's
pretrained head can be fine-tuned in place.
"""
import re
from pathlib import Path

import pandas as pd

# MELD's CSVs carry Windows-1252 punctuation as raw control characters
# (27% of utterances) - "you\x92re" tokenizes badly, "you're" doesn't.
_CP1252 = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"',
           "\x85": "...", "\x96": "-", "\x97": "-", "\xa0": " "}


def _clean_text(t):
    for k, v in _CP1252.items():
        t = t.replace(k, v)
    return re.sub(r"[\x80-\x9f]", "", t)


FER_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MELD_TO_FER = {"anger": "angry", "joy": "happy", "sadness": "sad"}
# Fixed emotion -> valence fallback (fusion learns a better mapping from logits)
EMOTION_VALENCE = {"neutral": 0, "angry": 1, "disgust": 1, "fear": 1, "sad": 1,
                   "happy": 2, "surprise": 2}


def load_split(processed, split, context=2, sep=" </s> "):
    processed = Path(processed)
    df = pd.read_csv(processed / f"manifest_{split}.csv")
    df["text"] = df.text.astype(str).map(_clean_text)
    if "speaker" not in df.columns:
        df["speaker"] = ""
    df["speaker"] = df.speaker.fillna("").astype(str).map(_clean_text)
    df["dia"] = df.key.str.extract(r"dia(\d+)_")[0].astype(int)
    df["utt"] = df.key.str.extract(r"utt(\d+)$")[0].astype(int)
    df = df.sort_values(["dia", "utt"]).reset_index(drop=True)

    parts = []
    for _, g in df.groupby("dia", sort=False):
        # Speaker-prefixed turns ("Monica: ...") let the encoder track who is
        # reacting to whom across the context window.
        turns = [f"{s}: {t}" if s else t
                 for s, t in zip(g.speaker.tolist(), g.text.astype(str).tolist())]
        g = g.copy()
        g["ctx_text"] = [sep.join(turns[max(0, i - context):i + 1])
                         for i in range(len(turns))]
        parts.append(g)
    df = pd.concat(parts).reset_index(drop=True)

    df["emotion_id"] = df.emotion.map(
        lambda e: FER_EMOTIONS.index(MELD_TO_FER.get(e, e)))
    df["wav"] = df.key.map(lambda k: str(processed / "audio" / split / f"{k}.wav"))
    face = df.key.map(lambda k: processed / "faces" / split / f"{k}.jpg")
    df["face"] = [str(p) if p.exists() else "" for p in face]
    return df
