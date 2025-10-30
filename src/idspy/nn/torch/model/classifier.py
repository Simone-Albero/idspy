from typing import Sequence, Optional, Callable, Tuple

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.mlp import MLPBlock
from ..module.embedding import EmbeddingBlock
from . import ModelFactory


class ModularClassifier(BaseModel):
    """Generic Classifier base class."""

    def __init__(
        self, feature_extractor: nn.Module, classifier_head: nn.Module
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier_head = classifier_head

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, features]

        Returns:
            Model output with 'logits' and 'latents'
        """
        latents = self.feature_extractor(x)
        logits = self.classifier_head(latents)
        return ModelOutput(logits=logits, latents=latents)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits and target for loss computation."""
        return output.logits, target


class ModularTabularClassifier(BaseModel):
    """Generic Classifier base class for tabular data."""

    def __init__(
        self,
        embedding_module: nn.Module,
        feature_extractor: nn.Module,
        classifier_head: nn.Module,
    ) -> None:
        super().__init__()
        self.embedding_module = embedding_module
        self.feature_extractor = feature_extractor
        self.classifier_head = classifier_head

    def forward(
        self, x_numerical: torch.Tensor, x_categorical: torch.Tensor
    ) -> ModelOutput:
        """Forward pass.

        Args:
            x_numerical: Tensor of shape [batch_size, n_numerical_features]
            x_categorical: Tensor of shape [batch_size, n_categorical_features]

        Returns:
            Model output with 'logits' and 'latents'
        """
        cat_emb = self.embedding(x_categorical)
        combined = torch.cat((x_numerical, cat_emb), dim=1)

        latents = self.feature_extractor(combined)
        logits = self.classifier_head(latents)

        return ModelOutput(logits=logits, latents=latents)

    def for_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits and target for loss computation."""
        return output.logits, target


@ModelFactory.register()
class MLPClassifier(ModularClassifier):
    """Two-stage MLP classifier with feature extraction and classification head."""

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
        hidden_dims = list(hidden_dims)

        # Feature extraction
        feat_dim = hidden_dims[-1] if hidden_dims else in_features
        extractor_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims

        feature_extractor = MLPBlock(
            in_features=in_features,
            out_features=feat_dim,
            hidden_dims=extractor_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

        # Classification head
        classifier_head = nn.Linear(feat_dim, out_features, bias=bias)

        super().__init__(
            feature_extractor=feature_extractor, classifier_head=classifier_head
        )


@ModelFactory.register()
class TabularClassifier(ModularTabularClassifier):
    """Classifier for mixed tabular data (numerical + categorical features)."""

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
        hidden_dims = list(hidden_dims)

        self.embedding = EmbeddingBlock(
            num_categorical, cat_cardinalities, max_emb_dim=max_emb_dim
        )
        emb_dim_total = sum(self.embedding.embedding_dims)

        in_features = num_numerical + emb_dim_total

        # Feature extraction
        feat_dim = hidden_dims[-1] if hidden_dims else in_features
        extractor_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims

        feature_extractor = MLPBlock(
            in_features=in_features,
            out_features=feat_dim,
            hidden_dims=extractor_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

        # Classification head
        classifier_head = nn.Linear(feat_dim, out_features, bias=bias)

        super().__init__(
            embedding_module=self.embedding,
            feature_extractor=feature_extractor,
            classifier_head=classifier_head,
        )
