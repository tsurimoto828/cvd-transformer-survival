from __future__ import annotations

from cvd_transformer_survival.utils import load_yaml


def test_config_files_have_required_sections(repo_root):
    train_config = load_yaml(repo_root / "configs" / "train.yaml")
    predict_config = load_yaml(repo_root / "configs" / "predict.yaml")

    assert set(["data", "model", "training", "output"]).issubset(train_config.keys())
    assert "categorical_columns" in train_config["data"]
    assert "continuous_columns" in train_config["data"]
    assert train_config["data"]["time_column"] == "time"
    assert train_config["data"]["event_column"] == "f"
    assert len(train_config["data"]["bins"]) == 11

    assert "inference" in predict_config
    assert predict_config["inference"]["metadata_filename"] == "metadata.json"
    assert predict_config["inference"]["id_column"] == "participant_id"
