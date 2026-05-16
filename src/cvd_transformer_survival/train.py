from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .trainer import train_nnet
from .utils import (
    build_metadata,
    drop_missing_rows,
    ensure_dir,
    fit_preprocessor,
    load_yaml,
    prepare_survival_dataframe,
    read_bins,
    save_json,
    set_seed,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TabTransformer survival model.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(int(config["training"].get("num_threads", 1)))

    train_path = Path(config["data"]["train_path"])
    valid_path = Path(config["data"]["valid_path"])
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    categorical_columns = config["data"].get("categorical_columns", [])
    continuous_columns = config["data"].get("continuous_columns", [])
    time_column = config["data"]["time_column"]
    event_column = config["data"]["event_column"]

    if config["data"].get("require_complete_cases", False):
        required_columns = list(categorical_columns) + list(continuous_columns) + [time_column, event_column]
        train_df = drop_missing_rows(train_df, required_columns)
        valid_df = drop_missing_rows(valid_df, required_columns)

    preprocessor = fit_preprocessor(train_df, categorical_columns, continuous_columns)
    train_data = prepare_survival_dataframe(train_df, preprocessor, time_column, event_column)
    valid_data = prepare_survival_dataframe(valid_df, preprocessor, time_column, event_column)

    bins = read_bins(config)

    device = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    params = {
        "device": device,
        "epochs": int(config["training"]["epochs"]),
        "batch_size": int(config["training"]["batch_size"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"].get("weight_decay", 0.0)),
        "patience": int(config["training"]["patience"]),
        "seed": seed,
        "categories": [len(preprocessor["categorical_levels"][col]) + 1 for col in categorical_columns],
        "num_continuous": len(continuous_columns),
        "dim": int(config["model"]["dim"]),
        "depth": int(config["model"]["depth"]),
        "heads": int(config["model"]["heads"]),
        "transformer_dropout": float(config["model"]["transformer_dropout"]),
        "mlp_hidden": int(config["model"].get("mlp_hidden", 64)),
        "mlp_hidden_layers": config["model"].get("mlp_hidden_layers"),
        "dropout": float(config["model"]["dropout"]),
        "pooling": config["model"].get("pooling", "cls"),
    }

    output_dir = ensure_dir(config["output"]["output_dir"])
    model_filename = config["output"].get("model_filename", "model_survival.pth")
    metadata_filename = config["output"].get("metadata_filename", "metadata.json")
    model_path = output_dir / model_filename
    log_filename = config["output"].get("log_filename", "training_log.csv")
    loss_plot_filename = config["output"].get("loss_plot_filename", "loss_curve.png")
    log_csv_path = output_dir / log_filename if log_filename else None
    loss_plot_path = output_dir / loss_plot_filename if loss_plot_filename else None

    model = train_nnet(
        train_data=train_data,
        valid_data=valid_data,
        bins=bins,
        params=params,
        model_save_path=model_path,
        log_csv_path=log_csv_path,
        loss_plot_path=loss_plot_path,
    )

    metadata = build_metadata(config, preprocessor, bins, params, model_filename=model_filename)
    save_json(output_dir / metadata_filename, metadata)

    summary = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "device": device,
        "model_path": str(model_path),
        "metadata_path": str(output_dir / metadata_filename),
        "training_log_path": str(log_csv_path) if log_csv_path else None,
        "loss_plot_path": str(loss_plot_path) if loss_plot_path else None,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    save_json(output_dir / "training_summary.json", summary)
    print("Training finished.")
    print(summary)


if __name__ == "__main__":
    main()
