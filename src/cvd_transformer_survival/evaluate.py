from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import load_yaml


def harrell_c_index(time, event, risk) -> dict[str, float]:
    """Compute Harrell's concordance index for right-censored survival data.

    Parameters
    ----------
    time:
        Observed follow-up time. Smaller values indicate earlier events/censoring.
    event:
        Event indicator, where 1 indicates an observed event and 0 indicates censoring.
    risk:
        Predicted risk score. Larger values indicate higher predicted risk.

    Returns
    -------
    dict
        Dictionary with c_index, concordant, discordant, tied_risk, comparable.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)

    valid = np.isfinite(time) & np.isfinite(risk) & ~pd.isna(event)
    time = time[valid]
    event = event[valid]
    risk = risk[valid]

    if len(time) == 0:
        raise ValueError("No valid rows were available for C-index calculation.")

    # Harrell's C-index compares each event case with subjects who survived longer.
    # Sorting by decreasing time lets a Fenwick tree hold risks for subjects with time > current time.
    order = np.argsort(-time, kind="mergesort")
    time = time[order]
    event = event[order]
    risk = risk[order]

    unique_risks = np.unique(risk)
    risk_rank = np.searchsorted(unique_risks, risk) + 1  # 1-indexed for Fenwick tree
    tree = np.zeros(len(unique_risks) + 2, dtype=np.int64)

    def add(idx: int, value: int = 1) -> None:
        while idx < len(tree):
            tree[idx] += value
            idx += idx & -idx

    def prefix_sum(idx: int) -> int:
        total = 0
        while idx > 0:
            total += int(tree[idx])
            idx -= idx & -idx
        return total

    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0
    comparable = 0.0
    n_at_longer_times = 0

    start = 0
    n = len(time)
    while start < n:
        end = start + 1
        while end < n and time[end] == time[start]:
            end += 1

        # The tree currently contains subjects with strictly longer follow-up times.
        for k in range(start, end):
            if event[k]:
                rank = int(risk_rank[k])
                lower = prefix_sum(rank - 1)
                equal = prefix_sum(rank) - prefix_sum(rank - 1)
                higher = n_at_longer_times - lower - equal

                concordant += lower
                tied_risk += equal
                discordant += higher
                comparable += n_at_longer_times

        # Add all subjects at the current time after scoring, excluding tied-time pairs.
        for k in range(start, end):
            add(int(risk_rank[k]), 1)
            n_at_longer_times += 1

        start = end

    if comparable == 0:
        raise ValueError(
            "No comparable pairs were available for C-index calculation. "
            "Check that at least one observed event has a shorter follow-up time than another subject."
        )

    c_index = (concordant + 0.5 * tied_risk) / comparable
    return {
        "c_index": float(c_index),
        "concordant": float(concordant),
        "discordant": float(discordant),
        "tied_risk": float(tied_risk),
        "comparable": float(comparable),
        "n_rows_used": float(len(time)),
    }


def _risk_from_predictions(
    predictions: pd.DataFrame,
    *,
    risk_column: Optional[str] = None,
    survival_column: Optional[str] = None,
    horizon_interval: Optional[int] = None,
) -> pd.Series:
    if risk_column:
        if risk_column not in predictions.columns:
            raise KeyError(f"risk_column '{risk_column}' was not found in prediction CSV.")
        return predictions[risk_column].astype(float)

    if survival_column:
        if survival_column not in predictions.columns:
            raise KeyError(f"survival_column '{survival_column}' was not found in prediction CSV.")
        return 1.0 - predictions[survival_column].astype(float)

    if horizon_interval is not None:
        col = f"cumulative_survival_interval_{horizon_interval}"
        if col not in predictions.columns:
            raise KeyError(
                f"Expected '{col}' in prediction CSV. "
                "Set evaluation.survival_column or evaluation.risk_column explicitly if needed."
            )
        return 1.0 - predictions[col].astype(float)

    cumulative_cols = [c for c in predictions.columns if c.startswith("cumulative_survival_interval_")]
    if not cumulative_cols:
        raise KeyError(
            "No risk column or cumulative survival columns were found. "
            "Set evaluation.risk_column, evaluation.survival_column, or evaluation.horizon_interval."
        )
    last_col = sorted(cumulative_cols, key=lambda x: int(x.rsplit("_", 1)[-1]))[-1]
    return 1.0 - predictions[last_col].astype(float)


def evaluate_from_config(config: dict) -> dict[str, float | str]:
    cfg = config["evaluation"]
    truth_path = Path(cfg["truth_path"])
    prediction_path = Path(cfg["prediction_path"])

    truth = pd.read_csv(truth_path)
    predictions = pd.read_csv(prediction_path)

    time_column = cfg.get("time_column", "time")
    event_column = cfg.get("event_column", "f")
    if time_column not in truth.columns:
        raise KeyError(f"time_column '{time_column}' was not found in truth CSV.")
    if event_column not in truth.columns:
        raise KeyError(f"event_column '{event_column}' was not found in truth CSV.")

    horizon = cfg.get("horizon")
    event = truth[event_column].astype(int).copy()
    time = truth[time_column].astype(float).copy()

    # For fixed-horizon evaluation, events occurring after the horizon are treated as censored at the horizon.
    if horizon is not None:
        horizon = float(horizon)
        event = ((event == 1) & (time <= horizon)).astype(int)
        time = time.clip(upper=horizon)

    risk = _risk_from_predictions(
        predictions,
        risk_column=cfg.get("risk_column"),
        survival_column=cfg.get("survival_column"),
        horizon_interval=cfg.get("horizon_interval"),
    )

    if len(truth) != len(predictions):
        id_column = cfg.get("id_column")
        if not id_column:
            raise ValueError(
                "truth CSV and prediction CSV have different row counts. "
                "Provide evaluation.id_column to align them."
            )
        if id_column not in truth.columns or id_column not in predictions.columns:
            raise KeyError(f"id_column '{id_column}' must exist in both CSV files.")
        merged = truth[[id_column, time_column, event_column]].merge(
            predictions[[id_column]].assign(_risk=risk), on=id_column, how="inner"
        )
        time = merged[time_column].astype(float)
        event = merged[event_column].astype(int)
        risk = merged["_risk"].astype(float)
        if horizon is not None:
            event = ((event == 1) & (time <= horizon)).astype(int)
            time = time.clip(upper=horizon)

    metrics = harrell_c_index(time=time, event=event, risk=risk)
    metrics.update(
        {
            "truth_path": str(truth_path),
            "prediction_path": str(prediction_path),
            "time_column": time_column,
            "event_column": event_column,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate survival predictions with Harrell's C-index.")
    parser.add_argument("--config", required=True, help="Path to evaluation YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    metrics = evaluate_from_config(config)

    output_path = config["evaluation"].get("output_path")
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(output_path, index=False)
        print(f"Evaluation metrics saved to {output_path}")

    print(f"Harrell's C-index: {metrics['c_index']:.6f}")
    print(f"Comparable pairs: {int(metrics['comparable'])}")


if __name__ == "__main__":
    main()
