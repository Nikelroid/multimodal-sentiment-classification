import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
    config.parse_cli_args()
    import os
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoImageProcessor
    from src.data.dataloaders import MultimodalDataset
    from src.data.preprocess import sent_preprocess
    from src.models.multimodal import MultimodalFusionNet
    from src.pipelines.train import collate_fn
    from src.configs import config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Evaluation on {device}...")
    
    test_dataset = MultimodalDataset(
        dataset_dir=config.data.msctd_dir,
        images_dir="dataset/test/test_ende",
        texts_file="dataset/test/english_test.txt",
        sentiments_file="dataset/test/sentiment_test.txt",
        preprocess_text_func=sent_preprocess,
        audio_dir="AudioSample"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    feature_extractor = AutoImageProcessor.from_pretrained(config.model.vision_backbone_name)
    collate = lambda b: collate_fn(b, tokenizer, feature_extractor)
    
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    
    model = MultimodalFusionNet(
        text_model_name=config.model.text_model_name,
        vit_model_name=config.model.vision_backbone_name,
        use_audio=True
    ).to(device)
    
    model_path = "models/best_multimodal.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded finalized checkpoint from {model_path}")
    else:
        print("Warning: No model checkpoint found. Expected 'models/best_multimodal.pt'")
        
    acc, p, r, f1, cm = evaluate_model(model, test_loader, device)
    
    print("-" * 30)
    print(f"Evaluation Accuracy : {acc:.4f}")
    print(f"Evaluation F1 Score : {f1:.4f}")
    print(f"Evaluation Precision: {p:.4f}")
    print(f"Evaluation Recall   : {r:.4f}")
    print("Confusion Matrix:\n", cm)
    print("-" * 30)
