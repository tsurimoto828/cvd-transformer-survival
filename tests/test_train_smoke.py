from __future__ import annotations

import json
import sys

from cvd_transformer_survival.train import main as train_main


def test_train_cli_smoke(monkeypatch, smoke_artifacts):
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(smoke_artifacts["train_yaml"])])
    train_main()

    artifact_dir = smoke_artifacts["artifact_dir"]
    model_path = artifact_dir / "model_survival.pth"
    metadata_path = artifact_dir / "metadata.json"
    summary_path = artifact_dir / "training_summary.json"

    assert model_path.exists()
    assert metadata_path.exists()
    assert summary_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert metadata["package_name"] == "cvd_transformer_survival"
    assert metadata["n_intervals"] == 5
    assert metadata["training_params"]["batch_size"] == 4
    assert summary["train_rows"] == len(smoke_artifacts["train_df"])
    assert summary["valid_rows"] == len(smoke_artifacts["valid_df"])
    assert summary["n_parameters"] > 0
