from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


CATEGORICAL_COLUMNS = [
    "sex",
    "urine_protein",
    "urine_glucose",
    "ecg_abnormality",
    "eye_exam",
    "anti_ht_med",
    "insulin_med",
    "anti_chol_med",
    "past_anemia",
    "smoker",
    "weight_gain",
    "regular_exercise",
    "daily_exercise",
    "fast_walking",
    "eating_speed",
    "late_meals",
    "skip_breakfast",
    "habitual_drinker",
    "alcohol_consumption",
    "enough_sleep",
    "lifestyle_change",
    "health_interest",
]

CONTINUOUS_COLUMNS = [
    "age",
    "height",
    "weight",
    "bmi",
    "waist",
    "sbp",
    "dbp",
    "rbc",
    "hemoglobin",
    "hematocrit",
    "ast",
    "alt",
    "ggt",
    "total_chol",
    "hdl_c",
    "triglyceride",
    "ldl_c",
    "glucose",
    "hba1c",
    "creatinine",
    "egfr",
]


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_survival_dataframe() -> pd.DataFrame:
    rows = []
    sexes = ["Male", "Female"]
    urine_three = ["(-)", "(±)", "(+)"]
    three_level = ["Normal", "Borderline abnormal", "Abnormal"]
    eating_speed = ["Slow", "Medium", "Fast"]
    alcohol = ["0g/day", "<20g/day", "<40g/day", "<60g/day", "≥60g/day"]
    lifestyle = ["No willingness", "Somewhat willing", "Willing"]

    for i in range(12):
        row = {
            "sex": sexes[i % 2],
            "urine_protein": urine_three[i % 3],
            "urine_glucose": urine_three[(i + 1) % 3],
            "ecg_abnormality": three_level[i % 3],
            "eye_exam": three_level[(i + 1) % 3],
            "anti_ht_med": "Yes" if i % 2 == 0 else "No",
            "insulin_med": "Yes" if i % 4 == 0 else "No",
            "anti_chol_med": "Yes" if i % 3 == 0 else "No",
            "past_anemia": "Yes" if i % 5 == 0 else "No",
            "smoker": "Yes" if i % 2 == 0 else "No",
            "weight_gain": "Yes" if i % 3 == 0 else "No",
            "regular_exercise": "Yes" if i % 2 == 1 else "No",
            "daily_exercise": "Yes" if i % 3 == 1 else "No",
            "fast_walking": "Yes" if i % 4 in (1, 2) else "No",
            "eating_speed": eating_speed[i % 3],
            "late_meals": "Yes" if i % 2 == 0 else "No",
            "skip_breakfast": "Yes" if i % 3 == 2 else "No",
            "habitual_drinker": "Yes" if i % 2 == 0 else "No",
            "alcohol_consumption": alcohol[i % 5],
            "enough_sleep": "Yes" if i % 2 == 1 else "No",
            "lifestyle_change": lifestyle[i % 3],
            "health_interest": "Yes" if i % 2 == 0 else "No",
            "age": 40 + i,
            "height": 160.0 + (i % 5),
            "weight": 55.0 + i,
            "bmi": 22.0 + 0.2 * i,
            "waist": 78.0 + i,
            "sbp": 118.0 + i,
            "dbp": 72.0 + (i % 6),
            "rbc": 430.0 + 3 * i,
            "hemoglobin": 13.0 + 0.1 * i,
            "hematocrit": 39.0 + 0.2 * i,
            "ast": 18.0 + i,
            "alt": 17.0 + i,
            "ggt": 22.0 + 2 * i,
            "total_chol": 190.0 + i,
            "hdl_c": 55.0 + (i % 4),
            "triglyceride": 110.0 + 3 * i,
            "ldl_c": 115.0 + i,
            "glucose": 90.0 + i,
            "hba1c": 5.3 + 0.05 * i,
            "creatinine": 0.65 + 0.01 * i,
            "egfr": 78.0 - 0.5 * i,
            "time": float((i % 5) + 1),
            "f": int(i % 2 == 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def smoke_artifacts(tmp_path: Path, tiny_survival_dataframe: pd.DataFrame):
    train_df = tiny_survival_dataframe.iloc[:8].reset_index(drop=True)
    valid_df = tiny_survival_dataframe.iloc[8:].reset_index(drop=True)
    test_df = valid_df.copy()
    test_df.insert(0, "participant_id", [f"P{i:03d}" for i in range(len(test_df))])
    test_df.loc[0, "eating_speed"] = "Very fast"  # unseen category to test fallback handling

    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    test_csv = tmp_path / "test.csv"
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.drop(columns=["time", "f"]).to_csv(test_csv, index=False)

    artifact_dir = tmp_path / "artifacts"
    output_predictions = tmp_path / "predictions" / "predictions.csv"

    train_config = {
        "seed": 0,
        "data": {
            "train_path": str(train_csv),
            "valid_path": str(valid_csv),
            "require_complete_cases": True,
            "time_unit": "years",
            "categorical_columns": CATEGORICAL_COLUMNS,
            "continuous_columns": CONTINUOUS_COLUMNS,
            "time_column": "time",
            "event_column": "f",
            "bins": [0, 1, 2, 3, 4, 5],
        },
        "model": {
            "dim": 16,
            "depth": 1,
            "heads": 4,
            "transformer_dropout": 0.1,
            "mlp_hidden_layers": [16, 8],
            "dropout": 0.1,
            "pooling": "cls",
        },
        "training": {
            "device": "cpu",
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 1.0e-5,
            "patience": 2,
            "num_threads": 1,
        },
        "output": {
            "output_dir": str(artifact_dir),
            "model_filename": "model_survival.pth",
            "metadata_filename": "metadata.json",
        },
    }

    predict_config = {
        "inference": {
            "artifact_dir": str(artifact_dir),
            "metadata_filename": "metadata.json",
            "test_path": str(test_csv),
            "output_path": str(output_predictions),
            "batch_size": 4,
            "device": "cpu",
            "num_threads": 1,
            "id_column": "participant_id",
        }
    }

    train_yaml = tmp_path / "train_smoke.yaml"
    predict_yaml = tmp_path / "predict_smoke.yaml"
    train_yaml.write_text(yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8")
    predict_yaml.write_text(yaml.safe_dump(predict_config, sort_keys=False), encoding="utf-8")

    return {
        "train_yaml": train_yaml,
        "predict_yaml": predict_yaml,
        "artifact_dir": artifact_dir,
        "test_df": test_df.drop(columns=["time", "f"]).copy(),
        "output_predictions": output_predictions,
        "train_df": train_df,
        "valid_df": valid_df,
    }
