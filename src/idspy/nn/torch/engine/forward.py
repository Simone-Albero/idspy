from typing import Union

import torch

from ..model.base import BaseModel, ModelOutput
from ....data.torch.batch import ensure_batch, Batch


def forward_pass(
    model: BaseModel, inputs: Union[torch.Tensor, Batch], device: torch.device
) -> ModelOutput:
    """Perform a forward pass through the model with the given inputs on the specified device.

    Args:
        model (BaseModel): The model to perform the forward pass.
        inputs (torch.Tensor or Batch): The input tensor to the model.
        device (torch.device): The device to run the model on.

    Returns:
        ModelOutput: The output from the model after the forward pass.
    """
    model.eval()
    inputs = ensure_batch(inputs).to(device, non_blocking=True)

    with torch.no_grad():
        outputs: ModelOutput = model(inputs)

    return outputs.detach().to(torch.device("cpu"))


def make_predictions(logits: torch.Tensor, pred_fn: callable) -> torch.Tensor:
    """Generate predictions from the model output using the provided prediction function.

    Args:
        logits (torch.Tensor): The output logits from the model.
        pred_fn (callable): A function that takes the model output and returns predictions.

    Returns:
        torch.Tensor: The predictions generated from the model output.
    """
    predictions = pred_fn(logits)
    return predictions
