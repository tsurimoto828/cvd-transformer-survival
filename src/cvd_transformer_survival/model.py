from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn



def split_x_batch(x_batch: torch.Tensor, num_categ: int, num_continuous: int) -> Tuple[torch.Tensor, torch.Tensor]:
    expected = num_categ + num_continuous
    if x_batch.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, but got {x_batch.shape[1]}.")

    x_categ = x_batch[:, :num_categ].long()
    x_numer = x_batch[:, num_categ:].float()
    return x_categ, x_numer


class NumericalEmbedder(nn.Module):
    def __init__(self, num_continuous: int, embed_dim: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_continuous, embed_dim))
        self.biases = nn.Parameter(torch.randn(num_continuous, embed_dim))

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        x_cont = x_cont.unsqueeze(-1)
        return x_cont * self.weights + self.biases


class TabTransformer(nn.Module):
    def __init__(
        self,
        categories: Sequence[int],
        num_continuous: int,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        use_cls: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.use_cls = use_cls
        self.pooling = pooling
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dim = dim

        self.category_embedders = nn.ModuleList([nn.Embedding(num_classes, dim) for num_classes in categories])
        self.numerical_embedder = NumericalEmbedder(num_continuous, dim)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm_pre = nn.LayerNorm(dim)
        self.norm_post = nn.LayerNorm(dim)
        self.attention_weights: List[torch.Tensor] = []

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        cat_embeds = [emb(x_categ[:, i]) for i, emb in enumerate(self.category_embedders)]
        x_categ_emb = torch.stack(cat_embeds, dim=1)
        x_cont_emb = self.numerical_embedder(x_cont)

        features = torch.cat([x_categ_emb, x_cont_emb], dim=1)
        batch_size = features.size(0)

        if self.use_cls:
            cls_token = self.cls_token.expand(batch_size, 1, -1)
            x = torch.cat([cls_token, features], dim=1)
        else:
            x = features

        x = self.norm_pre(x)
        x = self.transformer(x)
        x = self.norm_post(x)

        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "cls":
            return x[:, 0, :]
        raise ValueError(f"Invalid pooling method: {self.pooling}")

    def set_attention_tracking(self, enabled: bool) -> None:
        if enabled:
            raise NotImplementedError(
                "Attention map extraction is not included in the packaging-friendly training path. "
                "Run a separate analysis script if you need explicit attention maps."
            )
        self.attention_weights = []

    def get_attention_maps(self) -> List[torch.Tensor]:
        return self.attention_weights


class SurvivalModel(nn.Module):
    def __init__(
        self,
        categories: Sequence[int],
        num_continuous: int,
        n_intervals: int,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        transformer_dropout: float = 0.1,
        mlp_hidden: int = 64,
        mlp_hidden_layers: Sequence[int] | None = None,
        dropout: float = 0.3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.num_categ = len(categories)
        self.num_continuous = num_continuous

        self.tab_transformer = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            dropout=transformer_dropout,
            pooling=pooling,
        )

        hidden_layers = list(mlp_hidden_layers) if mlp_hidden_layers else [mlp_hidden]
        mlp_layers: List[nn.Module] = []
        in_dim = dim
        for hidden_dim in hidden_layers:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, n_intervals))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_categ, x_numer = split_x_batch(x_batch, self.num_categ, self.num_continuous)
        x = self.tab_transformer(x_categ, x_numer)
        return torch.sigmoid(self.mlp(x))
