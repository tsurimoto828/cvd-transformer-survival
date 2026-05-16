from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def save_json(path: str | Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def divide_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"time", "f"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required outcome columns: {sorted(missing)}")
    X = data.drop(columns=["time", "f"])
    y = data[["time", "f"]].copy()
    return X, y



def read_bins(config: Dict) -> np.ndarray:
    bins = config["data"].get("bins")
    if bins is None:
        raise ValueError("config.data.bins is required.")
    bins = np.asarray(bins, dtype=np.float32)
    if bins.ndim != 1 or len(bins) < 2:
        raise ValueError("Bins must be a 1D array with at least two break points.")
    if not np.all(np.diff(bins) > 0):
        raise ValueError("Bins must be strictly increasing.")
    return bins



def drop_missing_rows(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    return df.dropna(subset=list(required_columns)).copy()



def _fit_categorical_levels(series: pd.Series) -> List[str]:
    clean = series.astype("string").fillna("<NA>")
    unique_values = sorted(clean.unique().tolist())
    return unique_values



def fit_preprocessor(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
    continuous_columns: Sequence[str],
) -> Dict:
    missing = set(categorical_columns).union(continuous_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns in training data: {sorted(missing)}")

    categorical_levels = {
        col: _fit_categorical_levels(df[col]) for col in categorical_columns
    }

    continuous_means = {}
    continuous_stds = {}
    for col in continuous_columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        mean = float(numeric.mean()) if not numeric.dropna().empty else 0.0
        std = float(numeric.std(ddof=0)) if not numeric.dropna().empty else 1.0
        if np.isnan(std) or std == 0:
            std = 1.0
        continuous_means[col] = 0.0 if np.isnan(mean) else mean
        continuous_stds[col] = std

    return {
        "categorical_columns": list(categorical_columns),
        "continuous_columns": list(continuous_columns),
        "categorical_levels": categorical_levels,
        "continuous_means": continuous_means,
        "continuous_stds": continuous_stds,
    }



def transform_features(df: pd.DataFrame, preprocessor: Dict) -> pd.DataFrame:
    categorical_columns = preprocessor["categorical_columns"]
    continuous_columns = preprocessor["continuous_columns"]

    missing = set(categorical_columns).union(continuous_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns in input data: {sorted(missing)}")

    transformed = pd.DataFrame(index=df.index)

    for col in categorical_columns:
        levels = preprocessor["categorical_levels"][col]
        level_to_index = {level: idx for idx, level in enumerate(levels)}
        unknown_index = len(levels)
        clean = df[col].astype("string").fillna("<NA>")
        transformed[col] = clean.map(level_to_index).fillna(unknown_index).astype(int)

    for col in continuous_columns:
        mean = preprocessor["continuous_means"][col]
        std = preprocessor["continuous_stds"][col]
        numeric = pd.to_numeric(df[col], errors="coerce")
        transformed[col] = ((numeric - mean) / std).astype(float)

    return transformed[categorical_columns + continuous_columns]



def prepare_survival_dataframe(
    df: pd.DataFrame,
    preprocessor: Dict,
    time_column: str,
    event_column: str,
) -> pd.DataFrame:
    if time_column not in df.columns or event_column not in df.columns:
        raise ValueError(f"Input data must include '{time_column}' and '{event_column}'.")

    X = transform_features(df, preprocessor)
    y = pd.DataFrame(
        {
            "time": pd.to_numeric(df[time_column], errors="raise").astype(float),
            "f": pd.to_numeric(df[event_column], errors="raise").astype(int),
        },
        index=df.index,
    )
    return pd.concat([X, y], axis=1)



def prepare_features_only(df: pd.DataFrame, preprocessor: Dict) -> pd.DataFrame:
    return transform_features(df, preprocessor)



def build_metadata(config: Dict, preprocessor: Dict, bins: np.ndarray, params: Dict, model_filename: str) -> Dict:
    categories = [len(preprocessor["categorical_levels"][col]) + 1 for col in preprocessor["categorical_columns"]]
    metadata = {
        "package_name": "cvd_transformer_survival",
        "version": "0.1.0",
        "time_column": config["data"]["time_column"],
        "event_column": config["data"]["event_column"],
        "categorical_columns": preprocessor["categorical_columns"],
        "continuous_columns": preprocessor["continuous_columns"],
        "categorical_levels": preprocessor["categorical_levels"],
        "continuous_means": preprocessor["continuous_means"],
        "continuous_stds": preprocessor["continuous_stds"],
        "categories": categories,
        "num_continuous": len(preprocessor["continuous_columns"]),
        "bins": bins.tolist(),
        "n_intervals": len(bins) - 1,
        "model_params": {
            "dim": params["dim"],
            "depth": params["depth"],
            "heads": params["heads"],
            "transformer_dropout": params["transformer_dropout"],
            "mlp_hidden": params.get("mlp_hidden", 64),
            "mlp_hidden_layers": params.get("mlp_hidden_layers"),
            "dropout": params["dropout"],
            "pooling": params.get("pooling", "cls"),
        },
        "training_params": {
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params.get("weight_decay", 0.0),
            "epochs": params["epochs"],
            "patience": params["patience"],
            "seed": params.get("seed", 42),
        },
        "artifacts": {"model_filename": model_filename},
    }
    return metadata
