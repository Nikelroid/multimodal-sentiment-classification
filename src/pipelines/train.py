import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoImageProcessor
from src.configs import config
from src.data.dataloaders import MultimodalDataset
from src.data.collate import multimodal_collate
from src.models.multimodal import MultimodalFusionNet
from tqdm import tqdm
import wandb


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    """One pass over a loader. Trains when an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc)
    with torch.set_grad_enabled(training):
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            audio_values = batch["audio_values"].to(device) if batch["audio_values"] is not None else None
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(input_ids, attention_mask, pixel_values, audio_values)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                wandb.log({"batch_loss": loss.item()})

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / max(total, 1)})

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def train():
    import datetime
    run_name = f"fusion_{config.model.text_model_name.split('/')[-1]}_{config.model.vision_backbone_name.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

    wandb.init(
        name=run_name,
        entity=config.training.entity_name,
        project=config.training.project_name, config={
        "learning_rate": config.training.learning_rate,
        "epochs": config.training.max_epochs,
        "batch_size": config.training.batch_size,
        "text_model": config.model.text_model_name,
        "vision_model": config.model.vision_backbone_name,
        "use_audio": config.model.use_audio,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    feature_extractor = AutoImageProcessor.from_pretrained(config.model.vision_backbone_name)

    use_audio = config.model.use_audio
    collate = lambda b: multimodal_collate(b, tokenizer, feature_extractor,
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
    # drop_last avoids a size-1 final batch, which BatchNorm cannot normalize in train mode
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = MultimodalFusionNet(
        text_model_name=config.model.text_model_name,
        vit_model_name=config.model.vision_backbone_name,
        audio_model_name=config.model.audio_model_name,
        use_audio=use_audio,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float('inf')

    for epoch in range(config.training.max_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device,
                                          optimizer=optimizer, scaler=scaler,
                                          desc=f"Epoch {epoch+1}/{config.training.max_epochs}")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, desc="Validation")

        wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                   "val_loss": val_loss, "val_accuracy": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_multimodal.pt")
            print(f"Saved best model (val_loss={val_loss:.4f}).")

    wandb.finish()


if __name__ == "__main__":
    config.parse_cli_args()
    train()
