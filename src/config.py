import os
from pathlib import Path

# Paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"

# Datasets
MSCTD_DIR = DATA_DIR / "MSCTD"
INSTANY_DIR = DATA_DIR / "InstaNY100K"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
MAX_EPOCHS = 10
MAX_TEXT_LEN = 50

# Modern Models (Huggingface)
TEXT_MODEL_NAME = "roberta-base" # Modern robust NLP variant replacing generic BERT
VISION_BACKBONE_NAME = "google/vit-base-patch16-224-in21k" # ViT instead of plain EfficientNet optionally 
AUDIO_MODEL_NAME = "facebook/wav2vec2-base" # Standard audio processor
VISUALBERT_NAME = "uclanlp/visualbert-nlvr2"

# W&B Config
PROJECT_NAME = "multimodal-sentiment-classification"
ENTITY_NAME = os.getenv("WANDB_ENTITY", None)
