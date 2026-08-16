import torch


def multimodal_collate(batch, tokenizer, feature_extractor, max_text_len=None, use_audio=False):
    """Batch dict samples from MultimodalDataset into model-ready tensors."""
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]

    text_encodings = tokenizer(texts, padding=True, truncation=True,
                               max_length=max_text_len, return_tensors="pt")
    image_encodings = feature_extractor(images=images, return_tensors="pt")

    # Skip the audio stack entirely when audio is unused: the dummy waveforms
    # would otherwise be run through wav2vec2 for nothing.
    audio_values = torch.stack([item['audio'] for item in batch]) if use_audio else None

    return {
        "input_ids": text_encodings["input_ids"],
        "attention_mask": text_encodings["attention_mask"],
        "pixel_values": image_encodings["pixel_values"],
        "audio_values": audio_values,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
