from typing import Sequence, Optional, Callable, Union, Tuple

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.encoder import NumericalEncoder, CategoricalEncoder, TabularEncoder
from ..module.decoder import NumericalDecoder, CategoricalDecoder, TabularDecoder
from ....data.torch.batch import Features
from . import ModelFactory


@ModelFactory.register()
class Autoencoder(BaseModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Features) -> ModelOutput:
        """Forward pass.
        Args:
            x: Features object containing tensors or dictionary-like tensors.
        Returns:
            Model output with 'decoded' and 'latents'
        """
        encoded = self.encoder(**x)
        decoded = self.decoder(encoded)
        return ModelOutput(decoded=decoded, latents=encoded)

    def get_latent(self, x: Features) -> ModelOutput:
        """Get latent representation from encoder.
        Args:
            x: Features object containing tensors or dictionary-like tensors.
        Returns:
            Model output with 'latents'
        """
        encoded = self.encoder(**x)
        return ModelOutput(latents=encoded)

    def for_loss(
        self,
        output: ModelOutput,
        targets: Union[torch.Tensor, Features],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns reconstructed features and targets for loss computation."""
        return output["decoded"], targets


class NumericalAutoencoder(Autoencoder):

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

        super().__init__(encoder, decoder)


@ModelFactory.register()
class CategoricalAutoencoder(Autoencoder):
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

        super().__init__(encoder, decoder)


@ModelFactory.register()
class TabularAutoencoder(Autoencoder):

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

        super().__init__(encoder, decoder)

    def for_loss(
        self,
        output: ModelOutput,
        targets: Features,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns reconstructed features and targets for loss computation."""
        return (
            output["decoded"]["numerical"],
            output["decoded"]["categorical"],
            targets["numerical"],
            targets["categorical"],
        )
