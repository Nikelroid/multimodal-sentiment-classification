import torch.nn as nn
from transformers import AutoModel

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Lazy import: torchvision is only needed for this legacy backbone,
        # not for the default transformer pipeline.
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_b2(weights=weights)
        self.model.classifier = nn.Sequential() # Strip the classification head to get embeddings

    def forward(self, x):
        return self.model(x)

class ViTFeatureExtractor(nn.Module):
    """Generic vision-transformer feature extractor (ViT, DINOv2, SigLIP, ...)."""
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        if getattr(outputs, 'pooler_output', None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]  # CLS token
