import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from transformers import ViTModel

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_b2(weights=weights)
        self.model.classifier = nn.Sequential() # Strip the classification head to get embeddings

    def forward(self, x):
        return self.model(x)

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.pooler_output # (batch_size, hidden_size) generally 768
