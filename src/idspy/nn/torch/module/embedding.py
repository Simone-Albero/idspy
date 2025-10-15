from typing import Sequence, Optional

import torch
from torch import nn, Tensor


class EmbeddingBlock(nn.Module):
    """Feature embedding for categorical variables with padding/unknown support."""

    def __init__(
        self,
        num_features: Optional[int],
        cardinalities: Optional[Sequence[int]],
        max_emb_dim: int = 50,
    ):
        """Initialize feature embedding.

        Note:
            Index 0 is reserved for padding/unknown values and initialized to zeros.
        """
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.embedding_dims = []
        if num_features is None and cardinalities is None:
            raise ValueError("Either num_features or cardinalities must be provided.")

        if cardinalities is None:
            cardinalities = [max_emb_dim] * num_features  # Default cardinality

        for card in cardinalities:
            dim = min(max_emb_dim, int(card**0.5))
            emb = nn.Embedding(card + 1, dim, padding_idx=0)
            self.embeddings.append(emb)
            self.embedding_dims.append(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Categorical features [batch_size, num_categorical_features]
               Values with 0 are treated as padding/unknown

        Returns:
            Embedded features [batch_size, sum(embedding_dims)]
        """
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(x[:, i]))
        return torch.cat(embs, dim=1)

    def get_embedding_dims(self) -> Sequence[int]:
        """Get the list of embedding dimensions for each categorical feature."""
        return self.embedding_dims
