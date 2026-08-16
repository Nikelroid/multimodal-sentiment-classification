import torch
import torch.nn as nn
from transformers import WhisperModel


class AudioSentimentModel(nn.Module):
    """Speech-tone sentiment classifier: frozen Whisper encoder + small head.

    Uses a learnable softmax-weighted combination of ALL encoder layers
    (SUPERB-style): intermediate Whisper layers carry more prosodic/emotional
    information than the final layer, and the weighting lets training discover
    the right mix. Trained on RAVDESS + CREMA-D mapped to the MSCTD sentiment
    ids (neutral: 0, negative: 1, positive: 2) so outputs fuse directly with
    the multimodal model.
    """

    def __init__(self, model_name="openai/whisper-base", num_classes=3, freeze_encoder=True):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        n_layers = self.encoder.config.encoder_layers + 1  # + embedding output
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        d = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_features):
        # input_features: (B, n_mels, frames) log-mel from WhisperFeatureExtractor
        out = self.encoder(input_features, output_hidden_states=True)
        stacked = torch.stack(out.hidden_states, dim=0)          # (L+1, B, T, D)
        weights = torch.softmax(self.layer_weights, dim=0)
        mixed = (weights[:, None, None, None] * stacked).sum(0)  # (B, T, D)
        return self.head(mixed.mean(dim=1))
