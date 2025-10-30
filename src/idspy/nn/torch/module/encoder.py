from typing import Sequence, Optional, Callable, Union, Tuple

import torch
from torch import nn

from ..module.mlp import MLPBlock
from ..module.embedding import EmbeddingBlock


class NumericalEncoder(nn.Module):
    """Encoder for numerical features using MLP."""

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
            x: Tensor of shape [batch_size, n_numerical_features]
        Returns:
            Tensor of shape [batch_size, out_features]
        """
        encoded = self.encoder(x)
        return encoded


class CategoricalEncoder(nn.Module):
    """Encoder for categorical features using Embedding and MLP."""

    def __init__(
        self,
        out_features: int,
        num_categorical: Optional[int] = None,
        cat_cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.embedding = EmbeddingBlock(
            num_categorical, cat_cardinalities, max_emb_dim=max_emb_dim
        )

        embed_features = sum(self.embedding.get_embedding_dims())

        self.encoder = MLPBlock(
            in_features=embed_features,
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
    """Unified encoder for numerical and categorical features."""

    def __init__(
        self,
        num_numerical: int,
        out_features: int,
        num_categorical: Optional[int] = None,
        cat_cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities

        self.embedding = EmbeddingBlock(
            num_categorical, cat_cardinalities, max_emb_dim=max_emb_dim
        )

        embed_features = sum(self.embedding.get_embedding_dims())
        total_features = num_numerical + embed_features

        self.encoder = MLPBlock(
            in_features=total_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_numerical: Tensor of shape [batch_size, num_numerical]
            x_categorical: Tensor of shape [batch_size, n_cat_features]

        Returns:
            Tensor of shape [batch_size, out_features]
        """
        embedded_cat = self.embedding(x_categorical)

        combined = torch.cat([x_numerical, embedded_cat], dim=1)
        encoded = self.encoder(combined)

        return encoded
