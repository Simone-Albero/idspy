import torch


def make_predictions(pred_fn: callable, logits: torch.Tensor, *args) -> torch.Tensor:
    """Generate predictions from the model output using the provided prediction function.

    Args:
        pred_fn (callable): A function that takes the model output and returns predictions.
        logits (torch.Tensor): The output logits from the model.
        *args: Additional arguments to pass to the prediction function.

    Returns:
        torch.Tensor: The predictions generated from the model output.
    """
    predictions = pred_fn(logits, *args)
    return predictions
