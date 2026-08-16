"""Fine-tune the FER face model on MELD face crops (7-emotion labels).

MELD's emotion labels share the FER2013 inventory (after renaming
anger/joy/sadness), so the pretrained classification head is fine-tuned in
place - the model keeps its 7-emotion output for the playground while
adapting from posed FER2013 faces to real TV-show faces. Only utterances
with a detected face train; the fusion stage handles face-less utterances.

Usage: python src/pipelines/train_meld_face.py --processed /scratch1/.../processed
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.data.meld_dataset import EMOTION_VALENCE, FER_EMOTIONS, load_split
from src.pipelines.meld_common import evaluate, fit


class FaceDataset(Dataset):
    def __init__(self, df, mean, std, augment):
        df = df[df.face != ""].reset_index(drop=True)
        self.paths = df.face.tolist()
        self.emotions = df.emotion_id.tolist()
        self.sentiments = df.label.tolist()
        aug = [transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(0.2, 0.2, 0.1)] if augment else []
        self.tf = transforms.Compose(
            [*aug, transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return {"pixel_values": self.tf(img),
                "label": torch.tensor(self.emotions[i]),
                "sentiment": torch.tensor(self.sentiments[i])}


def forward(model, batch, device):
    return (model(pixel_values=batch["pixel_values"].to(device)).logits,
            batch["label"].to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--model_name", default="trpakov/vit-face-expression")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--out", default="models/meld_face.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    proc = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageClassification.from_pretrained(args.model_name).to(device)
    assert [model.config.id2label[i].lower() for i in range(7)] == FER_EMOTIONS

    loaders = []
    for split, train in [("train", True), ("dev", False), ("test", False)]:
        df = load_split(args.processed, split)
        ds = FaceDataset(df, proc.image_mean, proc.image_std, augment=train)
        print(f"{split}: {len(ds)} utterances with a face")
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                                  num_workers=6, pin_memory=True))

    # sqrt-inverse-frequency class weights: MELD emotion is heavily neutral
    # and unweighted training collapses the minority emotions (mF1 ~0.15).
    import numpy as np
    counts = np.bincount(loaders[0].dataset.emotions, minlength=7).astype(np.float64)
    weights = (counts.sum() / (7 * counts.clip(min=1))) ** 0.5
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    print(f"emotion counts {counts.tolist()} -> weights {[round(w, 2) for w in weights.tolist()]}")

    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids],
               "lr": args.lr},
              {"params": head, "lr": args.head_lr}]
    fit(model, forward, loaders, groups, args.out, device, epochs=args.epochs,
        class_weights=weights)

    # Secondary readout: emotion argmax -> valence vs the sentiment label
    # (fixed mapping; the fusion stage learns a better one from the logits).
    test = evaluate(model, loaders[2], forward, device)
    val_map = torch.tensor([EMOTION_VALENCE[e] for e in FER_EMOTIONS])
    mapped = val_map[torch.tensor(test["preds"])]
    sent = torch.tensor(loaders[2].dataset.sentiments)
    print(f"emotion->valence on test: acc {(mapped == sent).float().mean():.4f}")


if __name__ == "__main__":
    main()
