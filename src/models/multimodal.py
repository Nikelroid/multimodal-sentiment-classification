import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_models import ViTFeatureExtractor
from .text_models import TextFeatureExtractor
from .audio_models import AudioFeatureExtractor

class MultimodalFusionNet(nn.Module):
    """
    A unified multimodal fusion architecture supporting Text, Vision, and Audio inputs.
    It concats embeddings and passes through a classification head.
    """
    def __init__(self, text_model_name="roberta-base", vit_model_name="google/vit-base-patch16-224-in21k", audio_model_name="facebook/wav2vec2-base", num_classes=3, use_audio=False):
        super().__init__()
        self.use_audio = use_audio
        
        self.text_extractor = TextFeatureExtractor(text_model_name)
        self.image_extractor = ViTFeatureExtractor(vit_model_name)
        
        # Dimensions based on typical pretrained sizes (RoBERTa base = 768, ViT base = 768, wav2vec2 base = 768)
        self.text_dim = 768
        self.image_dim = 768
        self.audio_dim = 768 
        
        fused_dim = self.text_dim + self.image_dim
        
        if self.use_audio:
            self.audio_extractor = AudioFeatureExtractor(audio_model_name)
            fused_dim += self.audio_dim
            
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values, audio_values=None):
        txt_feats = self.text_extractor(input_ids, attention_mask)
        img_feats = self.image_extractor(pixel_values)
        
        features = [txt_feats, img_feats]
        
        if self.use_audio and audio_values is not None:
            aud_feats = self.audio_extractor(audio_values)
            features.append(aud_feats)
        elif self.use_audio and audio_values is None:
            # Fallback for missing audio in a batch but model supports it
            aud_feats = torch.zeros((txt_feats.shape[0], self.audio_dim)).to(txt_feats.device)
            features.append(aud_feats)
            
        fused = torch.cat(features, dim=1)
        logits = self.classifier(fused)
        return logits
