import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from src.configs import config
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
    import datetime
    run_name = f"fusion_{config.model.text_model_name.split('/')[-1]}_{config.model.vision_backbone_name.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    
    wandb.init(
        name=run_name,
        entity="kelidari-usc",
        project=config.training.project_name, config={
        "learning_rate": config.training.learning_rate,
        "epochs": config.training.max_epochs,
        "batch_size": config.training.batch_size,
        "text_model": config.model.text_model_name,
        "vision_model": config.model.vision_backbone_name
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    feature_extractor = AutoImageProcessor.from_pretrained(config.model.vision_backbone_name)

    # Wrap collation
    collate = lambda b: collate_fn(b, tokenizer, feature_extractor)

    # Initialize Dataset
    train_dataset = MultimodalDataset(
        dataset_dir=config.data.msctd_dir,
        images_dir="dataset/train/train_ende",
        texts_file="dataset/train/english_train.txt",
        sentiments_file="dataset/train/sentiment_train.txt",
        preprocess_text_func=sent_preprocess,
        audio_dir="AudioSample" # Will use blanks if not present
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    model = MultimodalFusionNet(
        text_model_name=config.model.text_model_name,
        vit_model_name=config.model.vision_backbone_name,
        use_audio=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')

    for epoch in range(config.training.max_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.max_epochs}")
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
    config.parse_cli_args()
    train()
