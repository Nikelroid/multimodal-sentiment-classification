import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import wandb

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            audio_values = batch["audio_values"].to(device) if batch["audio_values"] is not None else None
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask, pixel_values, audio_values)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro')
    r = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, p, r, f1, cm

def log_metrics_wandb(acc, p, r, f1, cm):
    # Ensure Wandb is initialized before calling this
    wandb.log({
        "eval_accuracy": acc,
        "eval_precision": p,
        "eval_recall": r,
        "eval_f1": f1
    })
    
    # Custom plotting via wandb depending on capabilities, 
    # but basic logs can suffice.
    print(f"Eval results - Acc: {acc}, F1: {f1}")

if __name__ == "__main__":
    pass # To be called within unified test notebook or scripts
