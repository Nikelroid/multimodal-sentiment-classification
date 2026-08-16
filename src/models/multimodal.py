import torch
import torch.nn as nn
from .image_models import ViTFeatureExtractor
from .text_models import TextFeatureExtractor
from .audio_models import AudioFeatureExtractor

class MultimodalFusionNet(nn.Module):
    """
    A unified multimodal fusion architecture supporting Text, Vision, optional
    Face-expression, and optional Audio inputs. It concats embeddings and
    passes through a classification head.
    """
    def __init__(self, text_model_name="roberta-base", vit_model_name="google/vit-base-patch16-224-in21k",
                 audio_model_name="facebook/wav2vec2-base", num_classes=3, use_audio=False,
                 use_face=False, face_model_name="trpakov/vit-face-expression"):
        super().__init__()
        self.use_audio = use_audio
        self.use_face = use_face

        self.text_extractor = TextFeatureExtractor(text_model_name)
        self.image_extractor = ViTFeatureExtractor(vit_model_name)

        # Embedding sizes come from the loaded backbones, so swapping e.g.
        # roberta-base -> roberta-large keeps the fusion head consistent.
        self.text_dim = self.text_extractor.encoder.config.hidden_size
        self.image_dim = self.image_extractor.encoder.config.hidden_size

        fused_dim = self.text_dim + self.image_dim

        if self.use_face:
            # Expression-pretrained ViT over the (precomputed) largest-face crop.
            self.face_extractor = ViTFeatureExtractor(face_model_name)
            self.face_dim = self.face_extractor.encoder.config.hidden_size
            fused_dim += self.face_dim
        else:
            self.face_dim = 0

        if self.use_audio:
            self.audio_extractor = AudioFeatureExtractor(audio_model_name)
            self.audio_dim = self.audio_extractor.encoder.config.hidden_size
            fused_dim += self.audio_dim
        else:
            self.audio_dim = 0

        # LayerNorm rather than BatchNorm: batch-size independent (works for
        # single-sample inference) and consistent with the transformer backbones.
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values, audio_values=None, face_values=None):
        txt_feats = self.text_extractor(input_ids, attention_mask)
        img_feats = self.image_extractor(pixel_values)

        features = [txt_feats, img_feats]

        if self.use_face:
            if face_values is not None:
                features.append(self.face_extractor(face_values))
            else:
                features.append(torch.zeros((txt_feats.shape[0], self.face_dim),
                                            device=txt_feats.device, dtype=txt_feats.dtype))

        if self.use_audio and audio_values is not None:
            features.append(self.audio_extractor(audio_values))
        elif self.use_audio and audio_values is None:
            # Fallback for missing audio in a batch but model supports it
            features.append(torch.zeros((txt_feats.shape[0], self.audio_dim),
                                        device=txt_feats.device, dtype=txt_feats.dtype))

        fused = torch.cat(features, dim=1)
        logits = self.classifier(fused)
        return logits
