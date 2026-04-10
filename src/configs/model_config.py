from dataclasses import dataclass

@dataclass
class ModelConfig:
    text_model_name: str = "roberta-base" # Modern robust NLP variant replacing generic BERT
    vision_backbone_name: str = "google/vit-base-patch16-224-in21k" # ViT replacing plain EfficientNet
    audio_model_name: str = "facebook/wav2vec2-base" # Standard audio processor
    visualbert_name: str = "uclanlp/visualbert-nlvr2"
    max_text_len: int = 50
