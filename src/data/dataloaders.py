import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

AUDIO_SAMPLES = 16000  # fixed 1-second window at 16 kHz so batches can always be stacked


class MultimodalDataset(Dataset):
    """
    A unified Multi-Modal Dataset supporting Image, Text, and optional Audio loading.
    Configurable to return specific modalities based on the model's need.
    """
    def __init__(self, dataset_dir, images_dir, texts_file, sentiments_file,
                 preprocess_text_func=None, image_transform=None, audio_dir=None,
                 audio_transform=None, audio_samples=AUDIO_SAMPLES, face_dir=None):

        self.dataset_path = Path(dataset_dir)
        self.images_path = self.dataset_path / images_dir
        self.sentiment_path = self.dataset_path / sentiments_file
        self.text_path = self.dataset_path / texts_file

        self.preprocess_text_func = preprocess_text_func
        self.image_transform = image_transform

        # Audio support (optional). Absolute paths are used as-is so the audio
        # directory does not have to live inside dataset_dir.
        if audio_dir:
            audio_dir = Path(audio_dir)
            self.audio_dir = audio_dir if audio_dir.is_absolute() else self.dataset_path / audio_dir
        else:
            self.audio_dir = None
        self.audio_transform = audio_transform
        self.audio_samples = audio_samples

        # Optional precomputed face crops (src/data/extract_faces.py); a
        # missing crop file means no face was detected in that frame.
        if face_dir:
            face_dir = Path(face_dir)
            self.face_dir = face_dir if face_dir.is_absolute() else self.dataset_path / face_dir
        else:
            self.face_dir = None

        with open(self.text_path, 'r') as f:
            self.texts = f.read().splitlines()

        with open(self.sentiment_path, 'r') as f:
            self.sentiments = np.array(f.read().splitlines()).astype("int64")

        if len(self.texts) != len(self.sentiments):
            print(f"Warning: {self.text_path.name} has {len(self.texts)} lines but "
                  f"{self.sentiment_path.name} has {len(self.sentiments)}; "
                  "using the smaller count.")
        self.length = min(len(self.texts), len(self.sentiments))

    def __len__(self):
        return self.length

    def _load_audio(self, idx):
        # Always return a fixed-length mono waveform (pad/truncate), otherwise
        # torch.stack in the collate function fails on mixed-length clips.
        audio_tensor = torch.zeros(self.audio_samples)
        if self.audio_dir is not None:
            audio_path = self.audio_dir / f"{idx}.wav"
            if os.path.exists(audio_path):
                try:
                    import librosa
                    waveform, _ = librosa.load(audio_path, sr=16000, mono=True,
                                               duration=self.audio_samples / 16000)
                    waveform = torch.tensor(waveform, dtype=torch.float32)
                    audio_tensor[:waveform.shape[0]] = waveform[:self.audio_samples]
                except Exception:
                    pass
        if self.audio_transform:
            audio_tensor = self.audio_transform(audio_tensor)
        return audio_tensor

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

        # 4. Face Modality (optional): blank crop when no face was detected
        face = None
        if self.face_dir is not None:
            try:
                face = Image.open(self.face_dir / f"{idx}.jpg").convert('RGB')
            except Exception:
                face = Image.new('RGB', (224, 224))

        return {
            "image": image,
            "text": text,
            "audio": self._load_audio(idx),
            "face": face,
            "label": int(self.sentiments[idx]),
        }
