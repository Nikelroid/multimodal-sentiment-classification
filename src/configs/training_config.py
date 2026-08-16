import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-5   # backbone LR
    head_lr: float = 1e-3         # fusion-head LR (randomly initialized -> larger)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.1
    max_epochs: int = 10
    patience: int = 3             # early-stopping patience on validation macro-F1
    num_workers: int = 2
    val_split: float = 0.1

    # W&B Config
    project_name: str = "multimodal-sentiment-classification"
    entity_name: str = os.getenv("WANDB_ENTITY", None)
