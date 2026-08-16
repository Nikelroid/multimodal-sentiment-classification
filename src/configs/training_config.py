import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-5
    max_epochs: int = 10
    num_workers: int = 2
    val_split: float = 0.1

    # W&B Config
    project_name: str = "multimodal-sentiment-classification"
    entity_name: str = os.getenv("WANDB_ENTITY", None)
