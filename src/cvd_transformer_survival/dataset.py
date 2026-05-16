from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SurvivalDataset(Dataset):
    def __init__(self, features: np.ndarray, y_surv: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()
        self.y_surv = torch.from_numpy(y_surv).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.y_surv[idx]


class InferenceDataset(Dataset):
    def __init__(self, features: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx]



def make_surv_array(t: np.ndarray, f: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    n_samples = t.shape[0]
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_samples, n_intervals * 2), dtype=np.float32)

    for i in range(n_samples):
        if f[i]:
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks[1:])
            if t[i] < breaks[-1]:
                y_train[i, n_intervals + np.where(t[i] < breaks[1:])[0][0]] = 1.0
        else:
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks_midpoint)
    return y_train



def prepare_dataloader(
    X,
    y,
    bins: np.ndarray,
    batch_size: int = 128,
    mode: str = "train",
    seed: Optional[int] = 42,
) -> DataLoader:
    y_surv = make_surv_array(y["time"].values, y["f"].values, bins)
    dataset = SurvivalDataset(X.values.astype(np.float32), y_surv)

    shuffle = mode == "train"
    drop_last = shuffle
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )



def prepare_inference_dataloader(X, batch_size: int = 128) -> DataLoader:
    dataset = InferenceDataset(X.values.astype(np.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
