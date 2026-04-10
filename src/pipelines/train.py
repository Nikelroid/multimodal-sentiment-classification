import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
import src.config
from src.data.dataloaders import MultimodalDataset
from src.data.preprocess import sent_preprocess
from src.models.multimodal import MultimodalFusionNet
from tqdm import tqdm
import wandb

def collate_fn(batch, tokenizer, feature_extractor):
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    audios = [item['audio'] for item in batch]
    labels = [item['label'] for item in batch]
    
    text_encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    image_encodings = feature_extractor(images=images, return_tensors="pt")
    
    return {
        "input_ids": text_encodings["input_ids"],
        "attention_mask": text_encodings["attention_mask"],
        "pixel_values": image_encodings["pixel_values"],
        "audio_values": torch.stack(audios) if audios[0] is not None else None,
        "labels": torch.tensor(labels, dtype=torch.long)
    }

def train():
    wandb.init(
        entity="kelidari-usc",
        project="multimodal-sentiment-analysis", config={
        "learning_rate": src.config.LEARNING_RATE,
        "epochs": src.config.MAX_EPOCHS,
        "batch_size": src.config.BATCH_SIZE,
        "text_model": src.config.TEXT_MODEL_NAME,
        "vision_model": src.config.VISION_BACKBONE_NAME
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(src.config.TEXT_MODEL_NAME)
    feature_extractor = AutoImageProcessor.from_pretrained(src.config.VISION_BACKBONE_NAME)

    # Wrap collation
    collate = lambda b: collate_fn(b, tokenizer, feature_extractor)

    # Initialize Dataset
    train_dataset = MultimodalDataset(
        dataset_dir=src.config.MSCTD_DIR,
        images_dir="dataset/train/train_ende",
        texts_file="dataset/train/english_train.txt",
        sentiments_file="dataset/train/sentiment_train.txt",
        preprocess_text_func=sent_preprocess,
        audio_dir="AudioSample" # Will use blanks if not present
    )
    
    train_loader = DataLoader(train_dataset, batch_size=src.config.BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=2)

    model = MultimodalFusionNet(
        text_model_name=src.config.TEXT_MODEL_NAME,
        vit_model_name=src.config.VISION_BACKBONE_NAME,
        use_audio=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=src.config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')

    for epoch in range(src.config.MAX_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{src.config.MAX_EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            audio_values = batch["audio_values"].to(device) if batch["audio_values"] is not None else None
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask, pixel_values, audio_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            wandb.log({"batch_loss": loss.item()})
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        wandb.log({"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_acc})
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_multimodal.pt")
            print("Saved best model.")

    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="classification", help="Task type")
    parser.add_argument("--data_dir", type=str, default=None, help="Base data directory")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name")
    parser.add_argument("--model_name", type=str, default=None, help="Override text model name")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    if args.data_dir:
        src.config.DATA_DIR = args.data_dir
    if args.model_name:
        src.config.TEXT_MODEL_NAME = args.model_name
    if args.batch_size:
        src.config.BATCH_SIZE = args.batch_size
        
    train()
