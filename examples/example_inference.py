from pathlib import Path

import pandas as pd

config_path = Path("configs/predict.yaml")
example_input_path = Path("examples/manuscript_example_input.csv")
pred_path = Path("results/manuscript_model/manuscript_example_predictions.csv")

if pred_path.exists():
    df = pd.read_csv(pred_path)
    if "cumulative_survival_interval_10" in df.columns:
        df["predicted_10y_cvd_risk"] = 1.0 - df["cumulative_survival_interval_10"]
    columns_to_show = [col for col in ["participant_id", "predicted_10y_cvd_risk", "cumulative_survival_interval_10"] if col in df.columns]
    print(df[columns_to_show].head())
else:
    print("Prediction output not found.")
    print("Expected example input:", example_input_path)
    if example_input_path.exists():
        template = pd.read_csv(example_input_path)
        print("
Example manuscript-aligned input preview:")
        print(template.head(2).to_string(index=False))
    print("
Run the manuscript-aligned inference command first:")
    print("python -m cvd_transformer_survival.predict --config configs/predict.yaml")
