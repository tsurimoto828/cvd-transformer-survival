from __future__ import annotations

import torch



def surv_likelihood_loss(n_intervals: int):
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        cens_uncens = 1.0 + y_true[:, :n_intervals] * (y_pred[:, :n_intervals] - 1.0)
        uncens = 1.0 - y_true[:, n_intervals:] * y_pred[:, :n_intervals]
        concatenated = torch.cat((cens_uncens, uncens), dim=-1)
        concatenated = torch.clamp(concatenated, min=1e-8)
        return -torch.log(concatenated).sum() / y_true.shape[0]

    return loss_fn
