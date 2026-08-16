"""Shared fp16 training loop for the MELD fine-tunes.

V100s have no bf16 hardware, so this uses fp16 autocast + GradScaler.
Early stopping and model selection use dev weighted-F1 (the standard MELD
reporting metric); the test split is evaluated once with the best model.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from transformers import get_cosine_schedule_with_warmup


def evaluate(model, loader, forward, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits, y = forward(model, batch, device)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return {"acc": float((preds == labels).mean()),
            "wf1": float(f1_score(labels, preds, average="weighted")),
            "mf1": float(f1_score(labels, preds, average="macro")),
            "preds": preds, "labels": labels}


def fit(model, forward, loaders, param_groups, out_path, device,
        epochs=6, patience=2, class_weights=None, label_smoothing=0.1):
    train_loader, dev_loader, test_loader = loaders
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total), total)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda")

    best_wf1, bad_epochs = -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, y = forward(model, batch, device)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch} step {step + 1}/{len(train_loader)} "
                      f"loss {running / (step + 1):.4f}", flush=True)

        dev = evaluate(model, dev_loader, forward, device)
        print(f"epoch {epoch}: train loss {running / len(train_loader):.4f} | "
              f"dev acc {dev['acc']:.4f} wF1 {dev['wf1']:.4f} mF1 {dev['mf1']:.4f}",
              flush=True)
        if dev["wf1"] > best_wf1:
            best_wf1, bad_epochs = dev["wf1"], 0
            torch.save(model.state_dict(), out_path)
            print(f"  saved {out_path} (dev wF1 {best_wf1:.4f})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("early stop", flush=True)
                break

    model.load_state_dict(torch.load(out_path, map_location=device))
    test = evaluate(model, test_loader, forward, device)
    print(f"TEST: acc {test['acc']:.4f} wF1 {test['wf1']:.4f} mF1 {test['mf1']:.4f}")
    print(confusion_matrix(test["labels"], test["preds"]))
    return test
