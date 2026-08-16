"""Late-fusion head over the MELD-fine-tuned modality models.

Caches, per utterance: probabilities AND penultimate features from the text
model (optionally two text backbones, e.g. RoBERTa-large + ModernBERT-large,
whose probabilities are also reported as an ensemble), the speech-tone model,
and the face model (probs/features averaged over the 25/50/75% frame crops
when available). Trains two fusion heads with modality dropout - probs-only
and full-features - picks the better on dev, and evaluates once on the MELD
test split. Presence flags keep it graceful when a modality is missing.

Writes models/meld_fusion.pt and models/meld_metrics.json.

Usage: python src/pipelines/train_meld_fusion.py --processed /scratch1/.../processed
"""
import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import soundfile as sf
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
BATCH = 24


@torch.no_grad()
def extract_text(df, ckpt, model_name, device, max_len):
    model = TextClassifier(model_name).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    tok = AutoTokenizer.from_pretrained(model_name)
    probs, feats = [], []
    for s in range(0, len(df), BATCH):
        enc = tok(df.ctx_text.iloc[s:s + BATCH].tolist(), truncation=True,
                  max_length=max_len, padding=True, return_tensors="pt")
        logits, pooled = model(enc["input_ids"].to(device),
                               enc["attention_mask"].to(device),
                               return_features=True)
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        feats.append(pooled.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(probs), np.concatenate(feats)


@torch.no_grad()
def extract_audio(df, ckpt, model_name, device):
    model = AudioSentimentModel(model_name=model_name).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    fe = WhisperFeatureExtractor.from_pretrained(model_name)
    probs, feats = [], []
    for s in range(0, len(df), BATCH):
        wavs = []
        for p in df.wav.iloc[s:s + BATCH]:
            wav, _ = sf.read(p, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            peak = float(np.abs(wav).max())
            wavs.append(wav * (0.95 / peak) if peak > 1e-6 else wav)
        f = fe(wavs, sampling_rate=16000, return_tensors="pt")["input_features"]
        af = model.features(f.to(device))
        probs.append(torch.softmax(model.head(af), 1).cpu().numpy())
        feats.append(af.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(probs), np.concatenate(feats)


@torch.no_grad()
def extract_face(df, ckpt, model_name, device, multi):
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    proc = AutoImageProcessor.from_pretrained(model_name)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(proc.image_mean, proc.image_std)])
    col = "face_multi" if multi else "face"
    n = len(df)
    probs, feats, has = np.zeros((n, 7)), None, np.zeros(n)
    for i, val in enumerate(df[col].tolist()):
        paths = [p for p in str(val).split(";") if p]
        if not paths:
            continue
        imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
        out = model(pixel_values=imgs.to(device), output_hidden_states=True)
        if feats is None:
            feats = np.zeros((n, out.hidden_states[-1].shape[-1]))
        probs[i] = torch.softmax(out.logits, 1).mean(0).cpu().numpy()
        feats[i] = out.hidden_states[-1][:, 0].mean(0).float().cpu().numpy()
        has[i] = 1.0
        if i % 1000 == 0:
            print(f"  face {i}/{n}", flush=True)
    del model
    torch.cuda.empty_cache()
    return probs, (feats if feats is not None else np.zeros((n, 768))), has


class FusionHead(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden),
                                 nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def pack(d, with_features, modality_dropout=0.0):
    text = [d["text"]] + ([d["tfeat"]] if with_features else [])
    if "text2" in d:
        text += [d["text2"]] + ([d["t2feat"]] if with_features else [])
    blocks = {"t": [b.copy() for b in text],
              "a": [b.copy() for b in
                    [d["audio"]] + ([d["afeat"]] if with_features else [])],
              "f": [b.copy() for b in
                    [d["face"]] + ([d["ffeat"]] if with_features else [])]}
    n = len(d["label"])
    flags = np.stack([np.ones(n), np.ones(n), d["has_face"].copy()], 1)
    if modality_dropout > 0:
        for i, k in enumerate(("t", "a", "f")):
            drop = np.random.rand(n) < modality_dropout
            for b in blocks[k]:
                b[drop] = 0.0
            flags[drop, i] = 0.0
    return np.concatenate(
        [*blocks["t"], *blocks["a"], *blocks["f"], flags], axis=1).astype(np.float32)


def train_head(data, y, device, with_features, hidden, epochs=200):
    d_in = pack(data["dev"], with_features).shape[1]
    head = FusionHead(d_in, hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    dev_x = torch.tensor(pack(data["dev"], with_features), device=device)
    best_wf1, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        x = torch.tensor(pack(data["train"], with_features, modality_dropout=0.15),
                         device=device)
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
    return head, best_wf1


def metrics(preds, labels):
    return {"acc": float((preds == labels).mean()),
            "wf1": float(f1_score(labels, preds, average="weighted")),
            "mf1": float(f1_score(labels, preds, average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--text_ckpt", default="models/meld_text.pt")
    ap.add_argument("--text_model_name", default="roberta-large")
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--text2_ckpt", default="")
    ap.add_argument("--text2_model_name", default="answerdotai/ModernBERT-large")
    ap.add_argument("--text2_context", type=int, default=8)
    ap.add_argument("--text2_max_len", type=int, default=320)
    ap.add_argument("--audio_ckpt", default="models/meld_audio.pt")
    ap.add_argument("--audio_model_name", default="openai/whisper-base")
    ap.add_argument("--face_ckpt", default="models/meld_face.pt")
    ap.add_argument("--face_model_name", default="trpakov/vit-face-expression")
    ap.add_argument("--multi_face", action="store_true")
    ap.add_argument("--cache_tag", default="v2")
    ap.add_argument("--out", default="models/meld_fusion.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    cache = {s: os.path.join(args.processed, f"fusion_feats_{args.cache_tag}_{s}.npz")
             for s in ("train", "dev", "test")}

    for split, path in cache.items():
        if os.path.exists(path):
            continue
        print(f"extracting {split} ...", flush=True)
        sep1 = f" {AutoTokenizer.from_pretrained(args.text_model_name).sep_token} "
        df = load_split(args.processed, split, context=args.context, sep=sep1)
        feats = {"label": df.label.to_numpy()}
        feats["text"], feats["tfeat"] = extract_text(
            df, args.text_ckpt, args.text_model_name, device, args.max_len)
        if args.text2_ckpt:
            sep2 = f" {AutoTokenizer.from_pretrained(args.text2_model_name).sep_token} "
            df2 = load_split(args.processed, split, context=args.text2_context, sep=sep2)
            assert (df2.key.to_numpy() == df.key.to_numpy()).all()
            feats["text2"], feats["t2feat"] = extract_text(
                df2, args.text2_ckpt, args.text2_model_name, device, args.text2_max_len)
        feats["audio"], feats["afeat"] = extract_audio(
            df, args.audio_ckpt, args.audio_model_name, device)
        feats["face"], feats["ffeat"], feats["has_face"] = extract_face(
            df, args.face_ckpt, args.face_model_name, device, args.multi_face)
        np.savez(path, **feats)

    data = {s: dict(np.load(p)) for s, p in cache.items()}
    y = {s: torch.tensor(d["label"], dtype=torch.long, device=device)
         for s, d in data.items()}

    head_p, dev_p = train_head(data, y, device, with_features=False, hidden=64)
    head_f, dev_f = train_head(data, y, device, with_features=True, hidden=256)
    print(f"dev wF1 - probs-only {dev_p:.4f} | with features {dev_f:.4f}")
    with_features = dev_f > dev_p
    head = head_f if with_features else head_p
    torch.save(head.state_dict(), args.out)

    report, d = {}, data["test"]
    labels = d["label"]
    report["text"] = metrics(d["text"].argmax(1), labels)
    if "text2" in d:
        report["text2"] = metrics(d["text2"].argmax(1), labels)
        report["text_ensemble"] = metrics(
            ((d["text"] + d["text2"]) / 2).argmax(1), labels)
    report["audio"] = metrics(d["audio"].argmax(1), labels)
    m = d["has_face"] == 1
    report["face_valence_on_faces"] = metrics(VAL_MAP[d["face"][m].argmax(1)], labels[m])
    report["face_coverage"] = float(m.mean())
    head.eval()
    with torch.no_grad():
        preds = head(torch.tensor(pack(d, with_features),
                                  device=device)).argmax(1).cpu().numpy()
    report["fused"] = metrics(preds, labels)
    report["fusion"] = {"chosen": "features" if with_features else "probs",
                       "dev_wf1_probs": float(dev_p), "dev_wf1_features": float(dev_f)}
    report["n_test"] = int(len(labels))
    print(json.dumps(report, indent=1))
    print(confusion_matrix(labels, preds))
    with open("models/meld_metrics.json", "w") as fjson:
        json.dump(report, fjson, indent=1)


if __name__ == "__main__":
    main()
