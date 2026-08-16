import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoImageProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from src.configs import config
from src.data.dataloaders import MultimodalDataset
from src.data.collate import multimodal_collate
from src.models.multimodal import MultimodalFusionNet
from tqdm import tqdm
import wandb


def build_optimizer(model, cfg):
    """Discriminative LRs: small for pretrained backbones, larger for the fusion
    head; weight decay skips biases and normalization parameters."""
    decay, no_decay, head_decay, head_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith("classifier")
        is_no_decay = p.ndim <= 1 or name.endswith(".bias")
        if is_head:
            (head_no_decay if is_no_decay else head_decay).append(p)
        else:
            (no_decay if is_no_decay else decay).append(p)
    groups = [
        {"params": decay, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": cfg.learning_rate, "weight_decay": 0.0},
        {"params": head_decay, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": head_no_decay, "lr": cfg.head_lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              scaler=None, amp_dtype=None, desc=""):
    """One pass over a loader. Trains when an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=desc)
    with torch.set_grad_enabled(training):
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            audio_values = batch["audio_values"].to(device) if batch["audio_values"] is not None else None
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(input_ids, attention_mask, pixel_values, audio_values)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                wandb.log({"batch_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / max(total, 1)})

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / max(len(loader), 1), correct / max(total, 1), macro_f1


def train():
    import datetime
    run_name = f"fusion_{config.model.text_model_name.split('/')[-1]}_{config.model.vision_backbone_name.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

    wandb.init(
        name=run_name,
        entity=config.training.entity_name,
        project=config.training.project_name, config={
        "learning_rate": config.training.learning_rate,
        "head_lr": config.training.head_lr,
        "epochs": config.training.max_epochs,
        "batch_size": config.training.batch_size,
        "label_smoothing": config.training.label_smoothing,
        "text_model": config.model.text_model_name,
        "vision_model": config.model.vision_backbone_name,
        "use_audio": config.model.use_audio,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    feature_extractor = AutoImageProcessor.from_pretrained(config.model.vision_backbone_name)

    use_audio = config.model.use_audio

    def collate(batch):
        return multimodal_collate(batch, tokenizer, feature_extractor,
                                  max_text_len=config.model.max_text_len,
                                  use_audio=use_audio)

    # Raw text goes straight to the tokenizer: transformer backbones perform
    # best without lowercasing/punctuation stripping.
    dataset = MultimodalDataset(
        dataset_dir=config.data.msctd_dir,
        images_dir="dataset/train/train_ende",
        texts_file="dataset/train/english_train.txt",
        sentiments_file="dataset/train/sentiment_train.txt",
        audio_dir=config.data.data_dir / "AudioSample" if use_audio else None,
    )

    # Held-out validation split for model selection.
    val_size = max(1, int(config.training.val_split * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    loader_args = dict(batch_size=config.training.batch_size, collate_fn=collate,
                       num_workers=config.training.num_workers,
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = MultimodalFusionNet(
        text_model_name=config.model.text_model_name,
        vit_model_name=config.model.vision_backbone_name,
        audio_model_name=config.model.audio_model_name,
        use_audio=use_audio,
    ).to(device)

    optimizer = build_optimizer(model, config.training)
    total_steps = len(train_loader) * config.training.max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.training.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

    # bf16 on Ampere+ needs no loss scaling; fall back to fp16 + GradScaler.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype, scaler = torch.bfloat16, None
    elif device.type == "cuda":
        amp_dtype, scaler = torch.float16, torch.amp.GradScaler("cuda")
    else:
        amp_dtype, scaler = None, None

    best_val_f1, epochs_without_improvement = -1.0, 0

    for epoch in range(config.training.max_epochs):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, amp_dtype=amp_dtype,
            desc=f"Epoch {epoch+1}/{config.training.max_epochs}")
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device, amp_dtype=amp_dtype, desc="Validation")

        wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                   "train_f1": train_f1, "val_loss": val_loss,
                   "val_accuracy": val_acc, "val_f1": val_f1})
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # Select on macro-F1: robust to class imbalance, unlike raw loss/accuracy.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_multimodal.pt")
            print(f"Saved best model (val_f1={val_f1:.4f}).")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.patience:
                print(f"Early stopping after {epoch+1} epochs (no val F1 gain "
                      f"for {config.training.patience}).")
                break

    wandb.finish()


if __name__ == "__main__":
    config.parse_cli_args()
    train()
