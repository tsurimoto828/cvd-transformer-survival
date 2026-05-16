from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .trainer import conditional_to_cumulative_survival, load_pytorch_model, predict_survival
from .utils import load_json, load_yaml, prepare_features_only



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained TabTransformer survival model.")
    parser.add_argument("--config", required=True, help="Path to prediction YAML config.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    torch.set_num_threads(int(config["inference"].get("num_threads", 1)))

    artifact_dir = Path(config["inference"]["artifact_dir"])
    metadata_filename = config["inference"].get("metadata_filename", "metadata.json")
    metadata = load_json(artifact_dir / metadata_filename)

    model_path = artifact_dir / metadata["artifacts"]["model_filename"]
    device = config["inference"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["inference"].get("batch_size", metadata["training_params"]["batch_size"]))

    params = {
        "device": device,
        "categories": metadata["categories"],
        "num_continuous": metadata["num_continuous"],
        "n_intervals": metadata["n_intervals"],
        **metadata["model_params"],
    }

    model = load_pytorch_model(params=params, model_path=model_path)

    test_path = Path(config["inference"]["test_path"])
    test_df = pd.read_csv(test_path)
    X_test = prepare_features_only(test_df, metadata)

    conditional_survival = predict_survival(model, X_test, batch_size=batch_size, device=device)
    cumulative_survival = conditional_to_cumulative_survival(conditional_survival)

    output_df = pd.DataFrame(index=test_df.index)
    id_column = config["inference"].get("id_column")
    if id_column and id_column in test_df.columns:
        output_df[id_column] = test_df[id_column]

    for i in range(conditional_survival.shape[1]):
        output_df[f"conditional_survival_interval_{i + 1}"] = conditional_survival[:, i]
        output_df[f"cumulative_survival_interval_{i + 1}"] = cumulative_survival[:, i]

    output_path = Path(config["inference"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
