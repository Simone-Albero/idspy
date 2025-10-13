from typing import Sequence, Optional, Callable, Union, Tuple

import torch
from torch import nn

from ..module.mlp import MLPBlock
from ..module.embedding import EmbeddingBlock


class NumericEncoder(nn.Module):
    """Encoder for numeric features using MLP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.encoder = MLPBlock(
            in_features=in_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, n_numeric_features]
        Returns:
            Tensor of shape [batch_size, out_features]
        """
        encoded = self.encoder(x)
        return encoded


class CategoricalEncoder(nn.Module):
    """Encoder for categorical features using Embedding and MLP."""

    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        embed_dim: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.embedding = EmbeddingBlock(
            cat_cardinalities=cat_cardinalities,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        in_features = embed_dim * len(cat_cardinalities)
        self.encoder = MLPBlock(
            in_features=in_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, n_cat_features]

        Returns:
            Tensor of shape [batch_size, out_features]
        """
        embedded = self.embedding(x)
        encoded = self.encoder(embedded)
        return encoded


class TabularEncoder(nn.Module):
    """Unified encoder for numeric and categorical features."""

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: Sequence[int],
        embed_dim: int,
        out_features: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if num_numeric <= 0 or len(cat_cardinalities) <= 0:
            raise ValueError(
                "Both num_numeric and cat_cardinalities must be > 0 for TabularEncoder"
            )

        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities

        self.embedding = EmbeddingBlock(
            cat_cardinalities=cat_cardinalities,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        embed_features = embed_dim * len(cat_cardinalities)
        total_features = num_numeric + embed_features

        self.encoder = MLPBlock(
            in_features=total_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_num: Tensor of shape [batch_size, num_numeric]
            x_cat: Tensor of shape [batch_size, n_cat_features]

        Returns:
            Tensor of shape [batch_size, out_features]
        """
        embedded_cat = self.embedding(x_cat)

        combined = torch.cat([x_num, embedded_cat], dim=1)
        encoded = self.encoder(combined)

        return encoded
