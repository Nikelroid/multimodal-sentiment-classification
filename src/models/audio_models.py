import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioFeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        
    def forward(self, input_values):
        # input_values: (batch, sequence_length)
        outputs = self.encoder(input_values)
        # Mean pooling over the sequence output
        hidden_states = outputs.last_hidden_state # (batch, seq_len, 768)
        return torch.mean(hidden_states, dim=1)
