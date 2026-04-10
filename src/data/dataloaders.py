import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F

class MultimodalDataset(Dataset):
    """
    A unified Multi-Modal Dataset supporting Image, Text, and optional Audio loading.
    Configurable to return specific modalities based on the model's need.
    """
    def __init__(self, dataset_dir, images_dir, texts_file, sentiments_file, 
                 preprocess_text_func=None, image_transform=None, audio_dir=None, audio_transform=None):
        
        self.dataset_path = Path(dataset_dir)
        self.images_path = self.dataset_path / images_dir
        self.sentiment_path = self.dataset_path / sentiments_file
        self.text_path = self.dataset_path / texts_file
        
        self.preprocess_text_func = preprocess_text_func
        self.image_transform = image_transform
        
        # Audio support (optional)
        self.audio_dir = self.dataset_path / audio_dir if audio_dir else None
        self.audio_transform = audio_transform

        with open(self.sentiment_path, 'r') as f:
            self.length = len(f.readlines())

        with open(self.text_path, 'r') as f:
            self.texts = f.read().splitlines()

        with open(self.sentiment_path, 'r') as f:
            self.sentiments = np.array(f.read().splitlines()).astype("int32")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Image Modality
        img_path = self.images_path / f'{idx}.jpg'
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback if image is missing
            image = Image.new('RGB', (224, 224))
            
        if self.image_transform:
            image = self.image_transform(image)
        
        # 2. Text Modality
        text = self.texts[idx].strip()
        if self.preprocess_text_func is not None:
            text = self.preprocess_text_func(text)
        
        # 3. Audio Modality (Optional, dummy if directory not present/matching)
        audio_tensor = torch.zeros((1, 16000)) # Dummy 1-sec silence
        if self.audio_dir and os.path.exists(self.audio_dir / f"{idx}.wav"):
            try:
                import librosa
                audio_path = self.audio_dir / f"{idx}.wav"
                waveform, sr = librosa.load(audio_path, sr=16000)
                audio_tensor = torch.tensor(waveform).unsqueeze(0)
                if self.audio_transform:
                    audio_tensor = self.audio_transform(audio_tensor)
            except Exception:
                pass
                
        sentiment = self.sentiments[idx]

        return {
            "image": image,
            "text": text,
            "audio": audio_tensor,
            "label": sentiment
        }
