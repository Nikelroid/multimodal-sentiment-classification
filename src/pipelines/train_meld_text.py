"""Fine-tune the text encoder on MELD utterance sentiment.

Each utterance is classified with its dialogue context (previous 2
utterances) prepended - context-free MELD text tops out several points
lower because short reactive utterances ("Yeah.", "Oh my God!") are
unreadable in isolation.

Usage: python src/pipelines/train_meld_text.py --processed /scratch1/.../processed
"""
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data.meld_dataset import load_split
from src.pipelines.meld_common import fit


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df.ctx_text.tolist()
        self.labels = df.label.tolist()
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "label": torch.tensor(self.labels[i])}


class TextClassifier(nn.Module):
    """Mask-aware mean pooling over the encoder + small head (matches the
    pooling used in the fusion model; HF checkpoints ship no pooler)."""

    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2),
                                  nn.Linear(d, 256), nn.GELU(),
                                  nn.Linear(256, num_classes))

    def forward(self, input_ids, attention_mask, return_features=False):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        logits = self.head(pooled)
        return (logits, pooled) if return_features else logits


def forward(model, batch, device):
    return (model(batch["input_ids"].to(device), batch["attention_mask"].to(device)),
            batch["label"].to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--model_name", default="roberta-large")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--out", default="models/meld_text.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    loaders = []
    for split, shuffle in [("train", True), ("dev", False), ("test", False)]:
        df = load_split(args.processed, split, context=args.context)
        loaders.append(DataLoader(TextDataset(df, tok, args.max_len),
                                  batch_size=args.batch_size, shuffle=shuffle,
                                  num_workers=4, pin_memory=True))

    model = TextClassifier(args.model_name).to(device)
    groups = [{"params": list(model.backbone.parameters()), "lr": args.lr},
              {"params": list(model.head.parameters()), "lr": args.head_lr}]
    fit(model, forward, loaders, groups, args.out, device,
        epochs=args.epochs)


if __name__ == "__main__":
    main()
