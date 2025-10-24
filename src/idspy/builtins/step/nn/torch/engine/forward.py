from typing import Any, Callable, Dict, Optional

import torch

from ......core.step.base import Step
from ......nn.torch.model.base import BaseModel, ModelOutput
from ......nn.torch.engine.forward import forward_pass, make_predictions
from .... import StepFactory


@StepFactory.register()
@Step.needs("inputs", "model", "device")
class ForwardOnce(Step):
    """Compute a single forward pass: model(input_tensor) -> output."""

    def __init__(
        self,
        to_cpu: bool = False,  # move output to CPU before storing
        inputs_key: str = "inputs",
        model_key: str = "model",
        device_key: str = "device",
        outputs_key: str = "outputs",
        name: Optional[str] = None,
    ) -> None:
        self.to_cpu = to_cpu

        super().__init__(name=name or "forward_once")

        self.key_map = {
            "inputs": inputs_key,
            "model": model_key,
            "device": device_key,
            "outputs": outputs_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, inputs: torch.Tensor, model: BaseModel, device: torch.device
    ) -> Optional[Dict[str, Any]]:
        out: ModelOutput = forward_pass(model, inputs, device)

        if self.to_cpu:
            out = out.detach().to(torch.device("cpu"))

        return {"outputs": out}


@StepFactory.register()
@Step.needs("logits", "labels")
class MakePredictions(Step):
    """Make predictions from model outputs."""

    def __init__(
        self,
        pred_fn: Callable,
        logits_key: str = "logits",
        outputs_key: str = "outputs",
        labels_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "make_predictions")
        self.pred_fn = pred_fn
        self.key_map = {
            "logits": logits_key,
            "outputs": outputs_key,
        }

        if labels_key is not None:
            self.key_map["labels"] = labels_key

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        predictions = (
            make_predictions(self.pred_fn, logits, labels).detach().cpu().numpy()
        )
        return {"outputs": predictions}
