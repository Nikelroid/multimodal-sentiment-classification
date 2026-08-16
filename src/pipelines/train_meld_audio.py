"""Fine-tune the speech-tone model on MELD audio sentiment.

Warm-starts from the RAVDESS/CREMA-D checkpoint (acted, studio-clean) and
adapts the layer mix + head to real sitcom audio - laugh track, music, and
overlapping speech included. The Whisper encoder stays frozen, so this is
cheap; only light gain augmentation is used because MELD audio is already
real-world messy.

Usage: python src/pipelines/train_meld_audio.py --processed /scratch1/.../processed
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from src.data.meld_dataset import load_split
from src.models.audio_sentiment import AudioSentimentModel
from src.pipelines.meld_common import fit


class MeldAudioDataset(Dataset):
    def __init__(self, df, feature_extractor, augment):
        self.paths = df.wav.tolist()
        self.labels = df.label.tolist()
        self.fe, self.augment = feature_extractor, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav, sr = sf.read(self.paths[i], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        peak = float(np.abs(wav).max())
        if peak > 1e-6:
            wav = wav * (0.95 / peak)      # match the serving-time normalization
        if self.augment:
            wav = wav * (10 ** np.random.default_rng().uniform(-0.75, 0.0))
        feats = self.fe(wav, sampling_rate=16000, return_tensors="np")["input_features"][0]
        return {"input_features": torch.tensor(feats),
                "label": torch.tensor(self.labels[i])}


def forward(model, batch, device):
    return model(batch["input_features"].to(device)), batch["label"].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--model_name", default="openai/whisper-base")
    ap.add_argument("--init", default="models/audio_sentiment.pt",
                    help="RAVDESS/CREMA-D checkpoint to warm-start from ('' = fresh)")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--unfreeze_top", type=int, default=2,
                    help="also fine-tune the top N Whisper encoder layers at 1e-5")
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--weight_power", type=float, default=0.5)
    ap.add_argument("--out", default="models/meld_audio.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    fe = WhisperFeatureExtractor.from_pretrained(args.model_name)
    loaders, train_df = [], None
    for split, train in [("train", True), ("dev", False), ("test", False)]:
        df = load_split(args.processed, split)
        if train:
            train_df = df
        loaders.append(DataLoader(MeldAudioDataset(df, fe, augment=train),
                                  batch_size=args.batch_size, shuffle=train,
                                  num_workers=6, pin_memory=True))

    model = AudioSentimentModel(model_name=args.model_name).to(device)
    if args.init and os.path.exists(args.init):
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"warm-started from {args.init}")

    counts = np.bincount(train_df.label, minlength=3).astype(np.float64)
    weights = (counts.sum() / (3 * counts)) ** args.weight_power
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    print(f"class counts {counts.tolist()} -> loss weights {weights.tolist()}")

    groups = [{"params": [model.layer_weights, *model.head.parameters()],
               "lr": args.lr}]
    if args.unfreeze_top > 0:
        # Domain adaptation: sitcom audio (laugh track, music, crosstalk) is
        # far from the acted studio corpora the warm-start saw.
        top = model.encoder.layers[-args.unfreeze_top:]
        for p in top.parameters():
            p.requires_grad = True
        groups.append({"params": list(top.parameters()), "lr": args.encoder_lr})
    fit(model, forward, loaders, groups,
        args.out, device, epochs=args.epochs, patience=3, class_weights=weights)


if __name__ == "__main__":
    main()
