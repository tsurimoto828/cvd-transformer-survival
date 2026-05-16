from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from cvd_transformer_survival.evaluate import harrell_c_index


def test_harrell_c_index_known_example():
    # Earlier event has higher risk than later subjects: fully concordant.
    metrics = harrell_c_index(
        time=[1.0, 2.0, 3.0, 4.0],
        event=[1, 1, 0, 0],
        risk=[0.9, 0.8, 0.2, 0.1],
    )
    assert metrics["c_index"] == 1.0
    assert metrics["comparable"] > 0


def test_evaluate_cli_after_prediction(repo_root: Path, smoke_artifacts):
    subprocess.run(
        [sys.executable, "-m", "cvd_transformer_survival.train", "--config", str(smoke_artifacts["train_yaml"])],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "cvd_transformer_survival.predict", "--config", str(smoke_artifacts["predict_yaml"])],
        cwd=repo_root,
        check=True,
    )

    truth_csv = smoke_artifacts["artifact_dir"].parent / "truth.csv"
    # Match the test rows used for prediction, including time/f columns.
    smoke_artifacts["valid_df"].to_csv(truth_csv, index=False)

    output_csv = smoke_artifacts["artifact_dir"].parent / "evaluation_metrics.csv"
    eval_config = {
        "evaluation": {
            "truth_path": str(truth_csv),
            "prediction_path": str(smoke_artifacts["output_predictions"]),
            "output_path": str(output_csv),
            "time_column": "time",
            "event_column": "f",
            "horizon": 5,
            "horizon_interval": 5,
        }
    }
    eval_yaml = smoke_artifacts["artifact_dir"].parent / "evaluate_smoke.yaml"
    eval_yaml.write_text(yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "cvd_transformer_survival.evaluate", "--config", str(eval_yaml)],
        cwd=repo_root,
        check=True,
    )

    metrics = pd.read_csv(output_csv)
    assert "c_index" in metrics.columns
    assert 0.0 <= float(metrics.loc[0, "c_index"]) <= 1.0
    assert float(metrics.loc[0, "comparable"]) > 0
