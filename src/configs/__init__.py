import argparse
import os
import yaml
from pathlib import Path
from typing import Optional

from .data_config import DataConfig
from .model_config import ModelConfig
from .training_config import TrainingConfig

class GlobalConfig:
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self._load_yaml()

    def _load_yaml(self, path="config.yml"):
        if os.path.exists(path):
            with open(path, "r") as f:
                y = yaml.safe_load(f)
                
            if y:
                # Merge Training
                if "training" in y:
                    for k, v in y["training"].items():
                        if hasattr(self.training, k):
                            setattr(self.training, k, v)
                
                # Merge Model
                if "model" in y:
                    for k, v in y["model"].items():
                        if hasattr(self.model, k):
                            setattr(self.model, k, v)
                            
                # Merge Data
                if "data" in y:
                    for k, v in y["data"].items():
                        if hasattr(self.data, k):
                            setattr(self.data, k, v)


    def parse_cli_args(self, args: Optional[list] = None):
        """
        Universal CLI argument parser that dynamically intercepts parameters
        across the entire pipeline and overrides respective dataclass properties.
        """
        parser = argparse.ArgumentParser(description="Multimodal Pipeline Configurations")
        
        # General / Data
        parser.add_argument("--task", type=str, default="classification", help="Task type")
        parser.add_argument("--data_dir", type=str, default=None, help="Base data directory override")
        parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name")
        
        # Model
        parser.add_argument("--model_name", type=str, default=None, help="Text model backbone name")
        parser.add_argument("--vision_model", type=str, default=None, help="Vision backbone name")
        
        # Training
        parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
        parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
        parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")

        parsed_args = parser.parse_args(args)

        # Propagate explicitly provided overrides securely
        if parsed_args.data_dir is not None:
            self.data.update_data_dir(parsed_args.data_dir)
            
        if parsed_args.model_name is not None:
            self.model.text_model_name = parsed_args.model_name
            
        if parsed_args.vision_model is not None:
            self.model.vision_backbone_name = parsed_args.vision_model
            
        if parsed_args.batch_size is not None:
            self.training.batch_size = parsed_args.batch_size
            
        if parsed_args.epochs is not None:
            self.training.max_epochs = parsed_args.epochs
            
        if parsed_args.learning_rate is not None:
            self.training.learning_rate = parsed_args.learning_rate

        return parsed_args

# Expose singleton
config = GlobalConfig()
