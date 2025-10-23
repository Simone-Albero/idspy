from typing import List, Tuple

from torch import Tensor
import torch
import torch.nn.functional as F

from .base import BaseLoss
from . import LossFactory


@LossFactory.register()
class NumericReconstructionLoss(BaseLoss):
    """Loss for reconstructing numerical features."""

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        """Initialize reconstruction loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__(reduction)

    def forward(
        self,
        out: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute reconstruction loss.

        Args:
            out: Reconstructed features [batch_size, num_features]
            target: Original features [batch_size, num_features]
        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        loss = F.mse_loss(out, target, reduction="none")
        loss = loss.mean(dim=-1)  # Average over features
        return self._reduce(loss)


@LossFactory.register()
class CategoricalReconstructionLoss(BaseLoss):
    """Loss for reconstructing categorical features."""

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        """Initialize reconstruction loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__(reduction)

    def _compute_loss_per_feature(
        self,
        recon: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute loss per categorical feature.
        Args:
            recon: List of logits tensors, one per categorical feature.
                   Each tensor has shape [batch_size, cardinality]
            targets: Tensor of shape [batch_size, n_cat_features] with true class indices
        Returns:
            Tensor of shape [batch_size, n_cat_features] with per-feature losses
        """

        losses = []
        for i, feat_logits in enumerate(recon):
            losses.append(
                F.cross_entropy(
                    feat_logits,
                    targets[:, i].long(),
                    reduction="none",
                )
            )

        return torch.stack(losses, dim=1)

    def forward(
        self,
        out: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute categorical reconstruction loss.

        Args:
            out: List of logits tensors, one per categorical feature.
                 Each tensor has shape [batch_size, cardinality]
            target: Tensor of shape [batch_size, n_cat_features] with true class indices

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        loss = self._compute_loss_per_feature(out, target)
        loss = loss.mean(dim=-1)  # Average over features
        return self._reduce(loss)


@LossFactory.register()
class TabularReconstructionLoss(BaseLoss):
    """Combined loss for reconstructing numeric and categorical features with learnable weighting."""

    def __init__(
        self,
        reduction: str = "mean",
        initial_alpha: float = 1.0,
        learnable_weight: bool = True,
    ) -> None:
        """Initialize tabular reconstruction loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            initial_alpha: Initial value for the categorical loss weight
            learnable_weight: Whether to make the weight learnable during training
        """
        super().__init__(reduction)

        self.numeric_loss = NumericReconstructionLoss(reduction="none")
        self.categorical_loss = CategoricalReconstructionLoss(reduction="none")

        if learnable_weight:
            # Use log-parameterization to ensure positive weights and better optimization
            self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(initial_alpha)))
        else:
            self.register_buffer("log_alpha", torch.log(torch.tensor(initial_alpha)))

        self.learnable_weight = learnable_weight

    @property
    def alpha(self) -> Tensor:
        """Get the current categorical loss weight."""
        return torch.exp(self.log_alpha)

    def forward(
        self,
        out_numerical: Tensor,
        out_categorical: List[Tensor],
        target_numerical: Tensor,
        target_categorical: Tensor,
    ) -> Tensor:
        """Compute combined reconstruction loss with learnable weighting.

        Args:
            out_numerical: Reconstructed numerical features [batch_size, num_numerical]
            out_categorical: List of logits tensors for each categorical feature
            target_numerical: Original numerical features [batch_size, num_numerical]
            target_categorical: Original categorical features [batch_size, n_cat_features]

        Returns:
            Weighted combination of numeric and categorical losses
        """
        numeric_loss = self.numeric_loss(out_numerical, target_numerical)
        categorical_loss = self.categorical_loss(out_categorical, target_categorical)

        # Use learnable weight for categorical loss
        loss = numeric_loss + self.alpha * categorical_loss
        return self._reduce(loss)

    def get_alpha_value(self) -> float:
        """Get the current value of alpha as a float."""
        return self.alpha.item()
