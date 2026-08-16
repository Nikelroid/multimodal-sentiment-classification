import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import wandb
from src.configs import config

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
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
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
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoImageProcessor
    from src.data.dataloaders import MultimodalDataset
    from src.data.collate import multimodal_collate
    from src.models.multimodal import MultimodalFusionNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Evaluation on {device}...")

    use_audio = config.model.use_audio
    test_dataset = MultimodalDataset(
        dataset_dir=config.data.msctd_dir,
        images_dir="dataset/test/test_ende",
        texts_file="dataset/test/english_test.txt",
        sentiments_file="dataset/test/sentiment_test.txt",
        audio_dir=config.data.data_dir / "AudioSample" if use_audio else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    feature_extractor = AutoImageProcessor.from_pretrained(config.model.vision_backbone_name)
    collate = lambda b: multimodal_collate(b, tokenizer, feature_extractor,
                                           max_text_len=config.model.max_text_len,
                                           use_audio=use_audio)

    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                             shuffle=False, collate_fn=collate,
                             num_workers=config.training.num_workers,
                             pin_memory=device.type == "cuda")

    model = MultimodalFusionNet(
        text_model_name=config.model.text_model_name,
        vit_model_name=config.model.vision_backbone_name,
        audio_model_name=config.model.audio_model_name,
        use_audio=use_audio,
    ).to(device)

    model_path = "models/best_multimodal.pt"
    if not os.path.exists(model_path):
        # Evaluating random weights produces misleading numbers, so refuse.
        sys.exit(f"Error: no checkpoint at '{model_path}'. Train one first: python src/pipelines/train.py")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded finalized checkpoint from {model_path}")

    acc, p, r, f1, cm = evaluate_model(model, test_loader, device)

    print("-" * 30)
    print(f"Evaluation Accuracy : {acc:.4f}")
    print(f"Evaluation F1 Score : {f1:.4f}")
    print(f"Evaluation Precision: {p:.4f}")
    print(f"Evaluation Recall   : {r:.4f}")
    print("Confusion Matrix:\n", cm)
    print("-" * 30)
