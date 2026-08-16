"""Late-fusion head over the three MELD-fine-tuned modality models.

Stage 1 caches each modality's probabilities for every utterance
(fusion_feats_<split>.npz). Stage 2 trains a small MLP on
[text probs(3) | audio probs(3) | face probs(7) | presence flags(3)]
with modality dropout, so the head learns per-modality reliability and
degrades gracefully when a modality is missing (about a quarter of MELD
mid-frames have no detectable face).

Evaluates each modality alone and the fusion on the MELD test split, and
writes models/meld_metrics.json.

Usage: python src/pipelines/train_meld_fusion.py --processed /scratch1/.../processed
"""
import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import transforms
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          AutoTokenizer, WhisperFeatureExtractor)

from src.data.meld_dataset import EMOTION_VALENCE, FER_EMOTIONS, load_split
from src.models.audio_sentiment import AudioSentimentModel
from src.pipelines.train_meld_text import TextClassifier

VAL_MAP = np.array([EMOTION_VALENCE[e] for e in FER_EMOTIONS])


@torch.no_grad()
def extract_split(df, models, device, batch_size=24):
    text_model, tok, audio_model, fe, face_model, face_tf = models
    n = len(df)
    feats = {"text": np.zeros((n, 3)), "audio": np.zeros((n, 3)),
             "face": np.zeros((n, 7)), "has_face": np.zeros(n),
             "label": df.label.to_numpy()}
    import soundfile as sf
    for s in range(0, n, batch_size):
        rows = df.iloc[s:s + batch_size]
        enc = tok(rows.ctx_text.tolist(), truncation=True, max_length=160,
                  padding=True, return_tensors="pt")
        logits = text_model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        feats["text"][s:s + len(rows)] = torch.softmax(logits, 1).cpu().numpy()

        wavs = []
        for p in rows.wav:
            wav, _ = sf.read(p, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            peak = float(np.abs(wav).max())
            wavs.append(wav * (0.95 / peak) if peak > 1e-6 else wav)
        f = fe(wavs, sampling_rate=16000, return_tensors="pt")["input_features"]
        feats["audio"][s:s + len(rows)] = torch.softmax(
            audio_model(f.to(device)), 1).cpu().numpy()

        imgs, idx = [], []
        for j, p in enumerate(rows.face):
            if p:
                imgs.append(face_tf(Image.open(p).convert("RGB")))
                idx.append(s + j)
        if imgs:
            logits = face_model(pixel_values=torch.stack(imgs).to(device)).logits
            feats["face"][idx] = torch.softmax(logits, 1).cpu().numpy()
            feats["has_face"][idx] = 1.0
        if (s // batch_size) % 50 == 0:
            print(f"  {s}/{n}", flush=True)
    return feats


class FusionHead(nn.Module):
    def __init__(self, d_in=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden),
                                 nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def pack(feats, modality_dropout=0.0):
    t, a, f = (feats["text"].copy(), feats["audio"].copy(), feats["face"].copy())
    flags = np.stack([np.ones(len(t)), np.ones(len(t)), feats["has_face"].copy()], 1)
    if modality_dropout > 0:
        for i, block in enumerate((t, a, f)):
            drop = np.random.rand(len(t)) < modality_dropout
            block[drop] = 0.0
            flags[drop, i] = 0.0
    return np.concatenate([t, a, f, flags], axis=1).astype(np.float32)


def metrics(preds, labels):
    return {"acc": float((preds == labels).mean()),
            "wf1": float(f1_score(labels, preds, average="weighted")),
            "mf1": float(f1_score(labels, preds, average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--text_ckpt", default="models/meld_text.pt")
    ap.add_argument("--audio_ckpt", default="models/meld_audio.pt")
    ap.add_argument("--face_ckpt", default="models/meld_face.pt")
    ap.add_argument("--text_model_name", default="roberta-large")
    ap.add_argument("--face_model_name", default="trpakov/vit-face-expression")
    ap.add_argument("--out", default="models/meld_fusion.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    cache = {s: os.path.join(args.processed, f"fusion_feats_{s}.npz")
             for s in ("train", "dev", "test")}

    if not all(os.path.exists(p) for p in cache.values()):
        text_model = TextClassifier(args.text_model_name).to(device).eval()
        text_model.load_state_dict(torch.load(args.text_ckpt, map_location=device))
        tok = AutoTokenizer.from_pretrained(args.text_model_name)
        audio_model = AudioSentimentModel().to(device).eval()
        audio_model.load_state_dict(torch.load(args.audio_ckpt, map_location=device))
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        face_model = AutoModelForImageClassification.from_pretrained(
            args.face_model_name).to(device).eval()
        face_model.load_state_dict(torch.load(args.face_ckpt, map_location=device))
        proc = AutoImageProcessor.from_pretrained(args.face_model_name)
        face_tf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(proc.image_mean, proc.image_std)])
        models = (text_model, tok, audio_model, fe, face_model, face_tf)
        for split, path in cache.items():
            print(f"extracting {split} ...", flush=True)
            feats = extract_split(load_split(args.processed, split), models, device)
            np.savez(path, **feats)
        del models, text_model, audio_model, face_model
        torch.cuda.empty_cache()

    data = {s: dict(np.load(p)) for s, p in cache.items()}
    y = {s: torch.tensor(d["label"], dtype=torch.long, device=device)
         for s, d in data.items()}

    head = FusionHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_wf1, best_state = -1.0, None
    dev_x = torch.tensor(pack(data["dev"]), device=device)
    for epoch in range(200):
        head.train()
        x = torch.tensor(pack(data["train"], modality_dropout=0.15), device=device)
        perm = torch.randperm(len(x), device=device)
        for s in range(0, len(x), 256):
            idx = perm[s:s + 256]
            opt.zero_grad()
            loss = criterion(head(x[idx]), y["train"][idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            preds = head(dev_x).argmax(1).cpu().numpy()
        wf1 = f1_score(data["dev"]["label"], preds, average="weighted")
        if wf1 > best_wf1:
            best_wf1 = wf1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    torch.save(head.state_dict(), args.out)
    print(f"fusion head saved (dev wF1 {best_wf1:.4f})")

    report, d = {}, data["test"]
    labels = d["label"]
    report["text"] = metrics(d["text"].argmax(1), labels)
    report["audio"] = metrics(d["audio"].argmax(1), labels)
    m = d["has_face"] == 1
    report["face_valence_on_faces"] = metrics(VAL_MAP[d["face"][m].argmax(1)], labels[m])
    report["face_coverage"] = float(m.mean())
    head.eval()
    with torch.no_grad():
        preds = head(torch.tensor(pack(d), device=device)).argmax(1).cpu().numpy()
    report["fused"] = metrics(preds, labels)
    report["n_test"] = int(len(labels))
    print(json.dumps(report, indent=1))
    print(confusion_matrix(labels, preds))
    with open("models/meld_metrics.json", "w") as fjson:
        json.dump(report, fjson, indent=1)


if __name__ == "__main__":
    main()
