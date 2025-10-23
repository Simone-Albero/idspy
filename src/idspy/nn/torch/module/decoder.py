from typing import Any, Sequence, Optional, Callable, Mapping

import torch
from torch import nn

from ..module.mlp import MLPBlock


class NumericalDecoder(nn.Module):
    """Decoder for numerical features using MLP."""

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

        self.decoder = MLPBlock(
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
            x: Tensor of shape [batch_size, latent_dim]

        Returns:
            Tensor of shape [batch_size, out_features]
        """
        decoded = self.decoder(x)
        return decoded


class CategoricalDecoder(nn.Module):
    """Decoder for categorical features using MLP and classification heads."""

    def __init__(
        self,
        in_features: int,
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

        self.cat_cardinalities = cat_cardinalities or [max_emb_dim] * num_categorical
        hidden_dims = list(hidden_dims)

        # Common decoder trunk
        if hidden_dims:
            self.decoder_trunk = MLPBlock(
                in_features=in_features,
                out_features=hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                activation=activation,
                norm_layer=norm_layer,
                dropout=dropout,
                bias=bias,
            )
            trunk_out_features = hidden_dims[-1]
        else:
            self.decoder_trunk = nn.Identity()
            trunk_out_features = in_features

        # Separate classification head for each categorical feature
        self.cat_heads = nn.ModuleList(
            [
                nn.Linear(trunk_out_features, cardinality)
                for cardinality in self.cat_cardinalities
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, latent_dim]

        Returns:
            List of logits tensors for each categorical feature, each of shape [batch_size, cardinality]
        """
        # Pass through common trunk
        decoded_features = self.decoder_trunk(x)

        # Generate logits for each categorical feature
        cat_logits = [head(decoded_features) for head in self.cat_heads]

        return cat_logits


class TabularDecoder(nn.Module):
    """Unified decoder for numerical and categorical features."""

    def __init__(
        self,
        in_features: int,
        num_numerical: int,
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

        self.cat_cardinalities = cat_cardinalities or [max_emb_dim] * num_categorical
        hidden_dims = list(hidden_dims)

        # Common decoder trunk
        if hidden_dims:
            self.decoder_trunk = MLPBlock(
                in_features=in_features,
                out_features=hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                activation=activation,
                norm_layer=norm_layer,
                dropout=dropout,
                bias=bias,
            )
            trunk_out_features = hidden_dims[-1]
        else:
            self.decoder_trunk = nn.Identity()
            trunk_out_features = in_features

        # Numeric features head
        self.numerical_head = nn.Linear(trunk_out_features, num_numerical)

        # Separate classification head for each categorical feature
        self.cat_heads = nn.ModuleList(
            [
                nn.Linear(trunk_out_features, cardinality)
                for cardinality in self.cat_cardinalities
            ]
        )

    def forward(self, x: torch.Tensor) -> Mapping[str, Any]:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, latent_dim]

        Returns:
            Dictionary with:
                - 'numerical': Tensor of shape [batch_size, num_numerical]
                - 'categorical': List of logits tensors for each categorical feature,each of shape [batch_size, cardinality]
        """

        decoded_features = self.decoder_trunk(x)

        # Generate numerical reconstructions
        reconstructed_numerical = self.numerical_head(decoded_features)

        # Generate logits for each categorical feature
        reconstructed_cat_logits = [head(decoded_features) for head in self.cat_heads]

        return {
            "numerical": reconstructed_numerical,
            "categorical": reconstructed_cat_logits,
        }
