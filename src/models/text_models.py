import torch.nn as nn
from transformers import AutoModel

class TextFeatureExtractor(nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mask-aware mean pooling over the sequence. Deliberately NOT
        # pooler_output: RoBERTa checkpoints ship without pooler weights, so the
        # pooler is a randomly-initialized tanh projection that wrecks the text
        # signal at fine-tuning LRs.
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
