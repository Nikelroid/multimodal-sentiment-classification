from pathlib import Path

from src.configs import GlobalConfig


def test_yaml_loads_from_repo_root_regardless_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # simulate launching from elsewhere
    cfg = GlobalConfig()
    assert cfg.model.text_model_name  # populated from config.yml, not defaults-only
    assert cfg.training.batch_size > 0


def test_data_dir_cascades():
    cfg = GlobalConfig()
    cfg.data.update_data_dir("/tmp/somewhere")
    assert cfg.data.msctd_dir == Path("/tmp/somewhere/MSCTD")
    assert cfg.data.instany_dir == Path("/tmp/somewhere/InstaNY100K")


def test_cli_overrides():
    cfg = GlobalConfig()
    cfg.parse_cli_args(["--batch_size", "8", "--epochs", "2", "--data_dir", "/tmp/d"])
    assert cfg.training.batch_size == 8
    assert cfg.training.max_epochs == 2
    assert cfg.data.data_dir == Path("/tmp/d")


def test_new_training_fields_exist():
    cfg = GlobalConfig()
    for field in ("head_lr", "weight_decay", "warmup_ratio", "label_smoothing",
                  "patience", "num_workers", "val_split"):
        assert hasattr(cfg.training, field)
    assert hasattr(cfg.model, "use_audio")
    assert hasattr(cfg.data, "dataset_name")
