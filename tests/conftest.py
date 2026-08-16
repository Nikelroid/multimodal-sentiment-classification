import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def fixture_dataset(tmp_path):
    """A tiny on-disk dataset in the exact MSCTD layout the loader expects."""
    images = tmp_path / "dataset" / "train" / "imgs"
    images.mkdir(parents=True)
    texts, sentiments = [], []
    for i in range(5):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images / f"{i}.jpg")
        texts.append(f"sample sentence number {i}")
        sentiments.append(str(i % 3))
    (tmp_path / "dataset" / "train" / "english_train.txt").write_text("\n".join(texts) + "\n")
    (tmp_path / "dataset" / "train" / "sentiment_train.txt").write_text("\n".join(sentiments) + "\n")
    return tmp_path
