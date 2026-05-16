from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .dataset import prepare_dataloader, prepare_inference_dataloader
from .loss import surv_likelihood_loss
from .model import SurvivalModel
from .utils import divide_data, ensure_dir


class ModelCheckpoint:
    def __init__(self, save_path: str | Path, monitor: str = "loss", mode: str = "min", patience: int = 5) -> None:
        self.save_path = str(save_path)
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, model: torch.nn.Module, epoch: int, metrics: Dict[str, float]) -> None:
        score = metrics[self.monitor]
        improved = (score < self.best_score) if self.mode == "min" else (score > self.best_score)

        if improved:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Model improved at epoch {epoch + 1}: {self.monitor} = {score:.4f}. Saved to {self.save_path}.")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {epoch + 1}.")

    def reset(self) -> None:
        self.best_score = float("inf") if self.mode == "min" else -float("inf")
        self.counter = 0
        self.early_stop = False



def train_nnet(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    bins: np.ndarray,
    params: Dict,
    model_save_path: str | Path = "results/models/model_survival.pth",
    log_csv_path: str | Path | None = None,
    loss_plot_path: str | Path | None = None,
) -> SurvivalModel:
    device = params["device"]
    epochs = params["epochs"]
    n_intervals = len(bins) - 1

    X_train, y_train = divide_data(train_data)
    X_valid, y_valid = divide_data(valid_data)

    train_loader = prepare_dataloader(X_train, y_train, bins, params["batch_size"], mode="train", seed=params.get("seed", 42))
    valid_loader = prepare_dataloader(X_valid, y_valid, bins, params["batch_size"], mode="test", seed=params.get("seed", 42))

    model = SurvivalModel(
        categories=params["categories"],
        num_continuous=params["num_continuous"],
        n_intervals=n_intervals,
        dim=params["dim"],
        depth=params["depth"],
        heads=params["heads"],
        transformer_dropout=params["transformer_dropout"],
        mlp_hidden=params.get("mlp_hidden", 64),
        mlp_hidden_layers=params.get("mlp_hidden_layers"),
        dropout=params["dropout"],
        pooling=params.get("pooling", "cls"),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 0.0),
    )
    criterion = surv_likelihood_loss(n_intervals)

    model_save_path = Path(model_save_path)
    ensure_dir(model_save_path.parent)
    callbacks = [ModelCheckpoint(model_save_path, monitor="loss", mode="min", patience=params["patience"])]
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_surv in train_loader:
            x_batch, y_surv = x_batch.to(device), y_surv.to(device)
            optimizer.zero_grad()
            out_surv = model(x_batch)
            loss = criterion(out_surv, y_surv)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x_batch, y_surv in valid_loader:
                x_batch, y_surv = x_batch.to(device), y_surv.to(device)
                out_surv = model(x_batch)
                loss = criterion(out_surv, y_surv)
                valid_loss += loss.item()
        valid_loss /= max(len(valid_loader), 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
        })

        metrics = {"loss": valid_loss}
        for cb in callbacks:
            cb(model, epoch, metrics)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        if any(cb.early_stop for cb in callbacks):
            print("Early stopping triggered.")
            break

    if log_csv_path is not None:
        log_csv_path = Path(log_csv_path)
        ensure_dir(log_csv_path.parent)
        pd.DataFrame(history).to_csv(log_csv_path, index=False)
        print(f"Training log saved to {log_csv_path}")

    if loss_plot_path is not None and len(history) > 0:
        loss_plot_path = Path(loss_plot_path)
        ensure_dir(loss_plot_path.parent)
        history_df = pd.DataFrame(history)
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
        plt.plot(history_df["epoch"], history_df["valid_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        print(f"Loss plot saved to {loss_plot_path}")

    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    return model



def load_pytorch_model(params: Dict, model_path: str | Path = "results/models/model_survival.pth") -> SurvivalModel:
    model_kwargs = {
        "categories": params["categories"],
        "num_continuous": params["num_continuous"],
        "n_intervals": params["n_intervals"],
        "dim": params["dim"],
        "depth": params["depth"],
        "heads": params["heads"],
        "transformer_dropout": params["transformer_dropout"],
        "mlp_hidden": params.get("mlp_hidden", 64),
        "mlp_hidden_layers": params.get("mlp_hidden_layers"),
        "dropout": params["dropout"],
        "pooling": params.get("pooling", "cls"),
    }
    model = SurvivalModel(**model_kwargs).to(params["device"])
    state_dict = torch.load(model_path, map_location=params["device"])
    model.load_state_dict(state_dict)
    return model



def predict_nnet(model: SurvivalModel, test_data: pd.DataFrame, bins: np.ndarray, params: Dict) -> np.ndarray:
    X_test, y_test = divide_data(test_data)
    test_loader = prepare_dataloader(X_test, y_test, bins, params["batch_size"], mode="test", seed=params.get("seed", 42))

    model.eval()
    surv_preds = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(params["device"])
            out_surv = model(x_batch)
            surv_preds.append(out_surv.cpu().numpy())

    return np.concatenate(surv_preds, axis=0)



def predict_survival(model: SurvivalModel, X_test: pd.DataFrame, batch_size: int, device: str) -> np.ndarray:
    test_loader = prepare_inference_dataloader(X_test, batch_size=batch_size)

    model.eval()
    surv_preds = []
    with torch.no_grad():
        for x_batch in test_loader:
            x_batch = x_batch.to(device)
            out_surv = model(x_batch)
            surv_preds.append(out_surv.cpu().numpy())

    return np.concatenate(surv_preds, axis=0)



def conditional_to_cumulative_survival(pred_surv: np.ndarray) -> np.ndarray:
    return np.cumprod(pred_surv, axis=1)
