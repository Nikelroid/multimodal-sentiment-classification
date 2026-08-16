import torch

from src.data.dataloaders import MultimodalDataset


def make_dataset(root, **kwargs):
    return MultimodalDataset(
        dataset_dir=root,
        images_dir="dataset/train/imgs",
        texts_file="dataset/train/english_train.txt",
        sentiments_file="dataset/train/sentiment_train.txt",
        **kwargs,
    )


def test_length_and_sample_keys(fixture_dataset):
    ds = make_dataset(fixture_dataset)
    assert len(ds) == 5
    sample = ds[0]
    assert set(sample) == {"image", "text", "audio", "label"}
    assert sample["label"] in (0, 1, 2)


def test_missing_image_falls_back_to_blank(fixture_dataset):
    (fixture_dataset / "dataset" / "train" / "imgs" / "3.jpg").unlink()
    ds = make_dataset(fixture_dataset)
    assert ds[3]["image"].size == (224, 224)


def test_audio_is_always_fixed_length(fixture_dataset):
    ds = make_dataset(fixture_dataset, audio_dir="no/such/dir")
    for i in range(len(ds)):
        assert ds[i]["audio"].shape == torch.Size([16000])


def test_text_sentiment_length_mismatch_uses_smaller_count(fixture_dataset):
    texts_file = fixture_dataset / "dataset" / "train" / "english_train.txt"
    texts_file.write_text(texts_file.read_text() + "an extra unlabeled line\n")
    ds = make_dataset(fixture_dataset)
    assert len(ds) == 5  # sentiment file still has 5 labels


def test_absolute_audio_dir_is_respected(fixture_dataset, tmp_path_factory):
    external = tmp_path_factory.mktemp("audio_elsewhere")
    ds = make_dataset(fixture_dataset, audio_dir=str(external))
    assert ds.audio_dir == external
