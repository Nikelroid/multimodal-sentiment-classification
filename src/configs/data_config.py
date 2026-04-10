import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    # Base paths
    base_dir: Path = field(default_factory=lambda: Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    data_dir: Path = None
    msctd_dir: Path = None
    instany_dir: Path = None

    def __post_init__(self):
        # Resolve data_dir
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        elif isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
            
        # Dynamically map sub-datasets
        self.msctd_dir = self.data_dir / "MSCTD"
        self.instany_dir = self.data_dir / "InstaNY100K"

    def update_data_dir(self, new_dir: str):
        """Update the base data directory and cascade to dependent paths."""
        self.data_dir = Path(new_dir)
        self.msctd_dir = self.data_dir / "MSCTD"
        self.instany_dir = self.data_dir / "InstaNY100K"
