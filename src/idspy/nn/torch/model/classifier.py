from typing import Sequence, Optional, Callable

import torch
from torch import nn

from .base import BaseModel, ModelOutput
from ..module.mlp import MLPBlock
from ..module.embedding import EmbeddingBlock
from ....data.torch.batch import Features
from . import ModelFactory


@ModelFactory.register()
class MLPClassifier(BaseModel):
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
        super().__init__()
        hidden_dims = list(hidden_dims)

        # Feature extraction
        feat_dim = hidden_dims[-1] if hidden_dims else in_features
        extractor_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims

        self.feature_extractor = MLPBlock(
            in_features=in_features,
            out_features=feat_dim,
            hidden_dims=extractor_dims,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
            bias=bias,
        )

        # Classification head
        self.classifier_head = nn.Linear(feat_dim, out_features, bias=bias)

    def forward(self, x: Features) -> ModelOutput:
        """Forward pass.

        Args:
            x: Features object containing tensor of shape [batch_size, features]

        Returns:
            Model output with 'logits' and 'latents'
        """
        latents = self.feature_extractor(x)
        logits = self.classifier_head(latents)
        return ModelOutput(logits=logits, latents=latents)


@ModelFactory.register()
class TabularClassifier(MLPClassifier):
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
        embedding = EmbeddingBlock(
            num_categorical, cat_cardinalities, max_emb_dim=max_emb_dim
        )
        emb_dim_total = sum(embedding.embedding_dims)

        in_features = num_numerical + emb_dim_total
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm_layer=norm_layer,
            bias=bias,
        )
        self.embedding = embedding

    def forward(self, x: Features) -> ModelOutput:
        """Forward pass.

        Args:
            x: Features object containing dict with 'numerical' and 'categorical' keys

        Returns:
            Model output with 'logits' and 'latents'
        """
        x_num = x["numerical"]
        x_cat = x["categorical"]

        cat_emb = self.embedding(x_cat)
        combined = torch.cat((x_num, cat_emb), dim=1)

        latents = self.feature_extractor(combined)
        logits = self.classifier_head(latents)

        return ModelOutput(logits=logits, latents=latents)
