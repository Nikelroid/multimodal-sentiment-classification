"""Train the speech-tone sentiment model on RAVDESS.

RAVDESS filenames encode the emotion in the third field:
01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust,
08 surprised. Mapped to MSCTD sentiment ids (neutral: 0, negative: 1,
positive: 2): {01, 02} -> neutral, {03, 08} -> positive,
{04, 05, 06, 07} -> negative.

Usage: python src/pipelines/train_audio.py --audio_dir /path/to/ravdess
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from src.models.audio_sentiment import AudioSentimentModel

EMOTION_TO_SENTIMENT = {1: 0, 2: 0, 3: 2, 8: 2, 4: 1, 5: 1, 6: 1, 7: 1}
VAL_ACTORS = {21, 22, 23, 24}          # RAVDESS speaker-independent val actors
CREMAD_EMOTIONS = {"NEU": 0, "HAP": 2, "ANG": 1, "DIS": 1, "FEA": 1, "SAD": 1}
CREMAD_VAL_MIN_ACTOR = 1080            # CREMA-D actors >= this go to validation


class RavdessDataset(Dataset):
    def __init__(self, files, feature_extractor):
        self.files = files
        self.fe = feature_extractor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = resample_poly(wav, 16000, sr).astype(np.float32)
        feats = self.fe(wav, sampling_rate=16000, return_tensors="np")["input_features"][0]
        return torch.tensor(feats), label


def scan(audio_dirs):
    """Collect (path, label) pairs from RAVDESS and/or CREMA-D style trees."""
    train, val = [], []
    for audio_dir in audio_dirs:
        for p in Path(audio_dir).rglob("*.wav"):
            dash, under = p.stem.split("-"), p.stem.split("_")
            if len(dash) == 7:                      # RAVDESS: 03-01-EMO-..-..-..-ACTOR
                emotion, actor = int(dash[2]), int(dash[6])
                item = (str(p), EMOTION_TO_SENTIMENT[emotion])
                (val if actor in VAL_ACTORS else train).append(item)
            elif len(under) == 4 and under[2] in CREMAD_EMOTIONS:  # CREMA-D: ID_SENT_EMO_LVL
                actor = int(under[0])
                item = (str(p), CREMAD_EMOTIONS[under[2]])
                (val if actor >= CREMAD_VAL_MIN_ACTOR else train).append(item)
    return train, val


def run(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, nargs="+",
                    help="one or more RAVDESS / CREMA-D roots")
    ap.add_argument("--model_name", default="openai/whisper-base")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/audio_sentiment.pt")
    ap.add_argument("--init", default=None, help="checkpoint to continue from")
    ap.add_argument("--unfreeze_top", type=int, default=0,
                    help="stage-2 fine-tuning: unfreeze the top N encoder layers")
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files, val_files = scan(args.audio_dir)
    print(f"train clips: {len(train_files)} | val clips (speaker-independent): {len(val_files)}")

    fe = WhisperFeatureExtractor.from_pretrained(args.model_name)
    train_ds, val_ds = RavdessDataset(train_files, fe), RavdessDataset(val_files, fe)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = AudioSentimentModel(args.model_name).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"initialized from {args.init}")
    if args.unfreeze_top > 0:
        for layer in model.encoder.layers[-args.unfreeze_top:]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f"unfroze top {args.unfreeze_top} encoder layers (lr={args.encoder_lr})")
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = list(model.head.parameters()) + [model.layer_weights]
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run(train_loader, model, criterion, device, optimizer)
        _, va_acc = run(val_loader, model, criterion, device)
        print(f"epoch {epoch+1}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            state = {k: (v.half() if v.is_floating_point() else v)
                     for k, v in model.state_dict().items()}
            torch.save(state, args.out)
            print(f"saved best (val_acc={va_acc:.4f})")
    print(f"BEST AUDIO VAL ACC: {best_acc:.4f}")


if __name__ == "__main__":
    main()
