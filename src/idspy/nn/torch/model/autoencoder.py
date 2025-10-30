from typing import Sequence, Optional, Callable, Tuple

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.encoder import NumericalEncoder, CategoricalEncoder, TabularEncoder
from ..module.decoder import NumericalDecoder, CategoricalDecoder, TabularDecoder
from . import ModelFactory


class ModularAutoencoder(BaseModel):
    """Generic Autoencoder with custom encoder and decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.
        Args:
            x: Input tensor
        Returns:
            Model output with 'decoded' and 'latents'
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return ModelOutput(decoded=decoded, latents=encoded)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns reconstructed features and targets for loss computation."""
        return output["decoded"], target


class ModularTabularAutoencoder(BaseModel):
    """Generic Tabular Autoencoder with custom encoder and decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> ModelOutput:
        """Forward pass.
        Args:
            x_numerical: Tensor of shape [batch_size, n_numerical_features]
            x_categorical: Tensor of shape [batch_size, n_categorical_features]
        Returns:
            Model output with 'decoded' and 'latents'
        """
        encoded = self.encoder(x_numerical, x_categorical)
        decoded = self.decoder(encoded)
        return ModelOutput(decoded=decoded, latents=encoded)

    def for_loss(
        self,
        output: ModelOutput,
        numerical_target: torch.Tensor,
        categorical_target: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns reconstructed features and targets for loss computation."""
        recon_numerical, recon_categorical = output["decoded"]

        return recon_numerical, recon_categorical, numerical_target, categorical_target


@ModelFactory.register()
class NumericalAutoencoder(ModularAutoencoder):

    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        hidden_dims = list(hidden_dims)

        encoder = NumericalEncoder(
            in_features=in_features,
            out_features=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )

        decoder = NumericalDecoder(
            in_features=latent_dim,
            out_features=in_features,
            hidden_dims=hidden_dims[::-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )

        super().__init__(encoder=encoder, decoder=decoder)


@ModelFactory.register()
class CategoricalAutoencoder(ModularAutoencoder):
    def __init__(
        self,
        embed_dim: int,
        latent_dim: int,
        num_categorical: Optional[int] = None,
        cat_cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        hidden_dims = list(hidden_dims)

        encoder = CategoricalEncoder(
            num_categorical=num_categorical,
            cat_cardinalities=cat_cardinalities,
            max_emb_dim=max_emb_dim,
            embed_dim=embed_dim,
            out_features=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )

        decoder = CategoricalDecoder(
            num_categorical=num_categorical,
            cat_cardinalities=cat_cardinalities,
            max_emb_dim=max_emb_dim,
            in_features=latent_dim,
            hidden_dims=hidden_dims[::-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )

        super().__init__(encoder=encoder, decoder=decoder)


@ModelFactory.register()
class TabularAutoencoder(ModularTabularAutoencoder):

    def __init__(
        self,
        num_numerical: int,
        latent_dim: int,
        num_categorical: Optional[int] = None,
        cat_cardinalities: Optional[Sequence[int]] = None,
        max_emb_dim: int = 50,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        hidden_dims = list(hidden_dims)

        encoder = TabularEncoder(
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            cat_cardinalities=cat_cardinalities,
            max_emb_dim=max_emb_dim,
            out_features=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )

        decoder = TabularDecoder(
            num_numerical=num_numerical,
            num_categorical=num_categorical,
            cat_cardinalities=cat_cardinalities,
            max_emb_dim=max_emb_dim,
            in_features=latent_dim,
            hidden_dims=hidden_dims[::-1],
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )

        super().__init__(encoder=encoder, decoder=decoder)
