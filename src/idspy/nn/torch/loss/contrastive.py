from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import BaseLoss
from . import LossFactory


@LossFactory.register()
class SupConLoss(BaseLoss):

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -1,
        tau: float = 0.07,
    ) -> None:
        """Initialize supervised contrastive loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
            tau: Temperature scaling factor
        """
        super().__init__(reduction)
        self.ignore_index = ignore_index
        self.tau = tau

    def forward(
        self,
        x: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute supervised contrastive loss.

        Args:
            x: Logits tensor [batch_size, num_classes]
            target: Target class indices [batch_size]

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        x = F.normalize(x, dim=1)
        N = x.shape[0]

        sim = (x @ x.T) / self.tau

        # Mask self-similarities to -inf
        mask_self = torch.eye(N, dtype=torch.bool, device=x.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        # Create positive mask based on target labels
        mask_pos = target.unsqueeze(0) == target.unsqueeze(1)
        mask_pos = mask_pos.masked_fill(mask_self, False)

        valid = target != self.ignore_index
        mask_pos = mask_pos & valid.unsqueeze(0) & valid.unsqueeze(1)

        # Count the number of positive pairs for each sample
        num_pos = mask_pos.sum(dim=1)

        valid_samples = num_pos > 0  # Samples with at least one positive pair
        if not valid_samples.any():
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        log_prob = F.log_softmax(sim, dim=1)
        log_prob_pos = (log_prob * mask_pos).sum(dim=1)  # sum over positives
        loss = -log_prob_pos[valid_samples] / num_pos[valid_samples]

        return self._reduce(loss)


@LossFactory.register()
class NtXentLoss(BaseLoss):

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -1,
        tau: float = 0.07,
    ) -> None:
        """Initialize NT-Xent loss. An unsupervised contrastive loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
            tau: Temperature scaling factor
        """
        super().__init__(reduction)
        self.ignore_index = ignore_index
        self.tau = tau

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        *args,
    ) -> Tensor:
        """Compute NT-Xent loss.

        Args:
            x1: First set of embeddings [batch_size, embedding_dim]
            x2: Second set of embeddings [batch_size, embedding_dim]

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)

        N = x1.shape[0]

        x = torch.cat([x1, x2], dim=0)  # [2*batch_size, embedding_dim]

        sim = (x @ x.T) / self.tau

        # Mask self-similarities to -inf
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=x.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        # Create positive mask for augmented pairs
        mask_pos = torch.zeros_like(sim, dtype=torch.bool)
        for i in range(N):
            mask_pos[i, i + N] = True
            mask_pos[i + N, i] = True

        log_prob = F.log_softmax(sim, dim=1)
        log_prob_pos = (log_prob * mask_pos).sum(dim=1)  # sum over positives

        loss = -log_prob_pos / 1  # each sample has one positive

        return self._reduce(loss)
