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
            Model output with 'recon' and 'latents'
        """
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        return ModelOutput(recon=recon, latents=encoded)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns reconstructed features and targets for loss computation."""
        return output["recon"], target


class ModularTabularAutoencoder(BaseModel):
    """Generic Tabular Autoencoder with custom encoder and decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        noise_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise_factor = noise_factor

    def _augment(
        self,
        x_numerical: torch.Tensor,
        x_categorical: torch.Tensor,
        noise_factor: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Data augmentation for contrastive learning.
        Args:
            x_numerical: Tensor of shape [batch_size, n_numerical_features]
            x_categorical: Tensor of shape [batch_size, n_categorical_features]
        Returns:
            Augmented numerical and categorical tensors.
        """
        # Gaussian noise to numerical features
        noise = torch.randn_like(x_numerical) * noise_factor
        x_numerical_aug = x_numerical + noise

        # Randomly mask some categorical features
        x_categorical_aug = x_categorical.clone()
        if x_categorical.numel() > 0:
            mask = torch.rand_like(x_categorical.float()) < noise_factor
            x_categorical_aug[mask] = 0  # Assuming 0 is the 'unknown' category

        return x_numerical_aug, x_categorical_aug

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> ModelOutput:
        """Forward pass.
        Args:
            x_numerical: Tensor of shape [batch_size, n_numerical_features]
            x_categorical: Tensor of shape [batch_size, n_categorical_features]
        Returns:
            Model output with 'recon' and 'latents'
        """
        if self.noise_factor > 0:
            x_numerical, x_categorical = self._augment(
                x_numerical, x_categorical, self.noise_factor
            )

        encoded = self.encoder(x_numerical, x_categorical)
        recon_numerical, recon_categorical = self.decoder(encoded)
        return ModelOutput(
            recon_numerical=recon_numerical,
            recon_categorical=recon_categorical,
            latents=encoded,
        )

    def for_loss(
        self,
        output: ModelOutput,
        target_numerical: torch.Tensor,
        target_categorical: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns reconstructed features and targets for loss computation."""
        recon_numerical, recon_categorical = (
            output["recon_numerical"],
            output["recon_categorical"],
        )

        return recon_numerical, recon_categorical, target_numerical, target_categorical


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
        noise_factor: float = 0.0,
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

        super().__init__(encoder=encoder, decoder=decoder, noise_factor=noise_factor)


@ModelFactory.register()
class ContrastiveTabularAutoencoder(ModularTabularAutoencoder):

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
        noise_factor: float = 0.1,
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

        super().__init__(encoder=encoder, decoder=decoder, noise_factor=noise_factor)

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> ModelOutput:
        """Forward pass with contrastive augmentations.
        Args:
            x_numerical: Tensor of shape [batch_size, n_numerical_features]
            x_categorical: Tensor of shape [batch_size, n_categorical_features]
        Returns:
            Model output with 'recon', 'latents', 'z1', and 'z2'
        """
        x1_num, x1_cat = self._augment(x_numerical, x_categorical, self.noise_factor)
        x2_num, x2_cat = self._augment(x_numerical, x_categorical, self.noise_factor)
        z1 = self.encoder(x1_num, x1_cat)
        z2 = self.encoder(x2_num, x2_cat)

        # Reconstruction (originale)
        z_orig = self.encoder(x_numerical, x_categorical)
        recon_num, recon_cat = self.decoder(z_orig)

        return ModelOutput(
            recon_numerical=recon_num,
            recon_categorical=recon_cat,
            latents=z_orig,
            z1=z1,
            z2=z2,
        )

    def for_loss(
        self,
        output: ModelOutput,
        target_numerical: torch.Tensor,
        target_categorical: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns augmented latents and reconstructions for loss computation."""

        recon_numerical, recon_categorical = (
            output["recon_numerical"],
            output["recon_categorical"],
        )

        return (
            output["z1"],
            output["z2"],
            recon_numerical,
            recon_categorical,
            target_numerical,
            target_categorical,
        )
