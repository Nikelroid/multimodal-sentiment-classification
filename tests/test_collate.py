import pytest
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from src.data.collate import multimodal_collate
from src.data.dataloaders import MultimodalDataset

# Small, fixed backbones for testing the collate contract (independent of the
# potentially much larger models configured in config.yml).
TEXT_MODEL = "roberta-base"
VISION_MODEL = "google/vit-base-patch16-224-in21k"


@pytest.fixture(scope="module")
def processors():
    return (AutoTokenizer.from_pretrained(TEXT_MODEL),
            AutoImageProcessor.from_pretrained(VISION_MODEL))


def load_batch(root, tokenizer, feature_extractor, **kwargs):
    ds = MultimodalDataset(
        dataset_dir=root,
        images_dir="dataset/train/imgs",
        texts_file="dataset/train/english_train.txt",
        sentiments_file="dataset/train/sentiment_train.txt",
    )
    return multimodal_collate([ds[i] for i in range(len(ds))],
                              tokenizer, feature_extractor, **kwargs)


def test_batch_shapes(fixture_dataset, processors):
    tok, fe = processors
    batch = load_batch(fixture_dataset, tok, fe, max_text_len=50)
    assert batch["input_ids"].shape[0] == 5
    assert batch["pixel_values"].shape == torch.Size([5, 3, 224, 224])
    assert batch["labels"].dtype == torch.long
    assert batch["audio_values"] is None  # audio disabled by default


def test_max_text_len_is_enforced(fixture_dataset, processors):
    tok, fe = processors
    long_text = fixture_dataset / "dataset" / "train" / "english_train.txt"
    long_text.write_text("word " * 500 + "\n" + "\n".join(["short"] * 4) + "\n")
    batch = load_batch(fixture_dataset, tok, fe, max_text_len=50)
    assert batch["input_ids"].shape[1] <= 50


def test_audio_stacking_when_enabled(fixture_dataset, processors):
    tok, fe = processors
    batch = load_batch(fixture_dataset, tok, fe, max_text_len=50, use_audio=True)
    assert batch["audio_values"].shape == torch.Size([5, 16000])
