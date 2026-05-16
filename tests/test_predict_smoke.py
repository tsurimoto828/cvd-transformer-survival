from __future__ import annotations

import sys

import pandas as pd

from cvd_transformer_survival.predict import main as predict_main
from cvd_transformer_survival.train import main as train_main


def test_predict_cli_smoke(monkeypatch, smoke_artifacts):
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(smoke_artifacts["train_yaml"])])
    train_main()

    monkeypatch.setattr(sys, "argv", ["predict", "--config", str(smoke_artifacts["predict_yaml"])])
    predict_main()

    output_path = smoke_artifacts["output_predictions"]
    assert output_path.exists()

    pred_df = pd.read_csv(output_path)
    test_df = smoke_artifacts["test_df"]

    assert len(pred_df) == len(test_df)
    assert pred_df.columns[0] == "participant_id"
    assert "conditional_survival_interval_1" in pred_df.columns
    assert "cumulative_survival_interval_5" in pred_df.columns

    conditional_cols = [c for c in pred_df.columns if c.startswith("conditional_survival_interval_")]
    cumulative_cols = [c for c in pred_df.columns if c.startswith("cumulative_survival_interval_")]

    assert pred_df[conditional_cols].ge(0.0).all().all()
    assert pred_df[conditional_cols].le(1.0).all().all()
    assert pred_df[cumulative_cols].ge(0.0).all().all()
    assert pred_df[cumulative_cols].le(1.0).all().all()

    diffs = pred_df[cumulative_cols].diff(axis=1).iloc[:, 1:]
    assert diffs.le(1e-8).all().all()
