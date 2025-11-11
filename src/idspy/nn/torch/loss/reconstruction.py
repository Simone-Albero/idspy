import logging
from pathlib import Path
from typing import Optional

from torch import Tensor
import torch
import torch.nn.functional as F

from .base import BaseLoss
from . import LossFactory

logger = logging.getLogger(__name__)


@LossFactory.register()
class NumericalReconstructionLoss(BaseLoss):
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
        x: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute reconstruction loss.

        Args:
            x: Reconstructed features [batch_size, num_features]
            target: Original features [batch_size, num_features]
        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        loss = F.mse_loss(x, target, reduction="none")
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

    def forward(
        self,
        x: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute categorical reconstruction loss.

        Args:
            x: Tensor of shape [batch_size, num_categorical, max_cardinality] with logits
            target: Tensor of shape [batch_size, num_categorical] with true class indices

        Returns:
            Loss tensor (scalar or per-sample based on reduction)
        """
        # Reshape x to [batch_size * num_categorical, max_cardinality]
        batch_size, num_categorical, max_cardinality = x.shape
        x_reshaped = x.view(-1, max_cardinality)

        # Reshape target to [batch_size * num_categorical]
        target_reshaped = target.view(-1).long()

        # Compute cross entropy loss
        loss = F.cross_entropy(x_reshaped, target_reshaped, reduction="none")

        # Reshape back to [batch_size, num_categorical]
        loss = loss.view(batch_size, num_categorical)

        # Average over features
        loss = loss.mean(dim=-1)

        return self._reduce(loss)


@LossFactory.register()
class TabularReconstructionLoss(BaseLoss):
    """Combined loss for reconstructing numeric and categorical features with learnable weighting."""

    def __init__(
        self,
        reduction: str = "mean",
        learnable_weight: bool = True,
        numerical_sigma: float = 1.0,
        categorical_sigma: float = 1.0,
        min_sigma: float = 0.1,
        max_sigma: float = 10.0,
        save_path: Optional[str] = None,
    ) -> None:
        """Initialize tabular reconstruction loss.

        Args:
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            learnable_weight: Whether to make the weights learnable during training
            numerical_sigma: Initial/fixed sigma (std) for numerical loss weighting
            categorical_sigma: Initial/fixed sigma (std) for categorical loss weighting
            min_sigma: Minimum allowed sigma value (prevents collapse to zero)
            max_sigma: Maximum allowed sigma value (prevents explosion)
            save_path: Path where to save/load learnable parameters (if learnable_weight=True)
        """
        super().__init__(reduction)

        self.numerical_loss = NumericalReconstructionLoss(reduction="none")
        self.categorical_loss = CategoricalReconstructionLoss(reduction="none")

        if learnable_weight:
            # Store sigma directly as learnable parameters
            self.numerical_sigma = torch.nn.Parameter(torch.tensor(numerical_sigma))
            self.categorical_sigma = torch.nn.Parameter(torch.tensor(categorical_sigma))
        else:
            # Store as fixed buffers
            self.register_buffer("numerical_sigma", torch.tensor(numerical_sigma))
            self.register_buffer("categorical_sigma", torch.tensor(categorical_sigma))

        self.learnable_weight = learnable_weight
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.save_path = Path(save_path) if save_path else None

        if self.save_path and self.learnable_weight:
            try:
                self.load_parameters()
            except Exception as e:
                logger.warning(
                    f"Could not load parameters from {self.save_path}, "
                    f"using standard initialization. Error: {e}"
                )

    def forward(
        self,
        x_numerical: Tensor,
        x_categorical: Tensor,
        target_numerical: Tensor,
        target_categorical: Tensor,
    ) -> Tensor:
        """Compute combined reconstruction loss with uncertainty-based weighting.

        Args:
            x_numerical: Reconstructed numerical features [batch_size, num_numerical]
            x_categorical: Tensor of shape [batch_size, num_categorical, max_cardinality] with logits
            target_numerical: Original numerical features [batch_size, num_numerical]
            target_categorical: Original categorical features [batch_size, num_categorical]

        Returns:
            Weighted combination of numeric and categorical losses with uncertainty regularization
        """
        numerical_loss = self.numerical_loss(x_numerical, target_numerical)
        categorical_loss = self.categorical_loss(x_categorical, target_categorical)

        # Clamp sigmas to prevent collapse or explosion
        numerical_sigma = torch.clamp(
            self.numerical_sigma, self.min_sigma, self.max_sigma
        )
        categorical_sigma = torch.clamp(
            self.categorical_sigma, self.min_sigma, self.max_sigma
        )

        # Compute log-variance from sigma: log(σ²) = 2*log(σ)
        log_var_numerical = 2 * torch.log(numerical_sigma)
        log_var_categorical = 2 * torch.log(categorical_sigma)

        # Homoscedastic uncertainty weighting
        # Loss = (1/2σ²) * L + (1/2) * log(σ²)
        numerical_precision = torch.exp(-log_var_numerical)
        categorical_precision = torch.exp(-log_var_categorical)

        loss = (
            numerical_precision * numerical_loss
            + categorical_precision * categorical_loss
            + 0.5 * log_var_numerical
            + 0.5 * log_var_categorical
        )

        return self._reduce(loss)

    def get_weights(self) -> dict:
        """Get the current learned weights."""
        # Use clamped values for reporting
        numerical_sigma = torch.clamp(
            self.numerical_sigma, self.min_sigma, self.max_sigma
        )
        categorical_sigma = torch.clamp(
            self.categorical_sigma, self.min_sigma, self.max_sigma
        )

        # Compute log-variance for reporting
        log_var_numerical = 2 * torch.log(numerical_sigma)
        log_var_categorical = 2 * torch.log(categorical_sigma)

        return {
            "numerical_sigma": numerical_sigma.item(),
            "categorical_sigma": categorical_sigma.item(),
            "numerical_log_variance": log_var_numerical.item(),
            "categorical_log_variance": log_var_categorical.item(),
            "numerical_precision": torch.exp(-log_var_numerical).item(),
            "categorical_precision": torch.exp(-log_var_categorical).item(),
        }

    def save_parameters(self, path: Optional[str] = None) -> None:
        """Save learnable parameters to disk.

        Args:
            path: Path where to save parameters. If None, uses self.save_path
        """
        if not self.learnable_weight:
            logger.warning("No learnable parameters to save")
            return

        save_path = Path(path) if path else self.save_path

        if save_path is None:
            logger.warning("No save path provided, skipping parameter save")
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        weights = self.get_weights()
        torch.save(
            {
                "numerical_sigma": self.numerical_sigma.data,
                "categorical_sigma": self.categorical_sigma.data,
                "weights_info": weights,
            },
            save_path,
        )

        logger.info(f"Saved learnable parameters to {save_path}")

    def load_parameters(self, path: Optional[str] = None) -> None:
        """Load learnable parameters from disk.

        Args:
            path: Path from where to load parameters. If None, uses self.save_path
        """
        if not self.learnable_weight:
            logger.warning(
                "Cannot load parameters: loss is not using learnable weights"
            )
            return

        load_path = Path(path) if path else self.save_path

        if load_path is None:
            logger.warning("No load path provided, skipping parameter load")
            return

        if not load_path.exists():
            logger.error(f"Parameter file not found at {load_path}")
            return

        try:
            checkpoint = torch.load(load_path, map_location="cpu")

            # Load the sigma parameters
            self.numerical_sigma.data = checkpoint["numerical_sigma"]
            self.categorical_sigma.data = checkpoint["categorical_sigma"]

        except Exception as e:
            logger.error(f"Failed to load parameters from {load_path}: {e}")
            raise

    def __del__(self):
        if self.learnable_weight:
            weights = self.get_weights()
            logger.info(
                f"Final learned weights:"
                f"\n  - Numerical sigma: {weights['numerical_sigma']:.4f}"
                f"\n  - Categorical sigma: {weights['categorical_sigma']:.4f}"
                f"\n  - Numerical precision: {weights['numerical_precision']:.4f}"
                f"\n  - Categorical precision: {weights['categorical_precision']:.4f}"
            )

            if self.save_path:
                try:
                    self.save_parameters()
                except Exception as e:
                    logger.error(f"Failed to auto-save parameters: {e}")
