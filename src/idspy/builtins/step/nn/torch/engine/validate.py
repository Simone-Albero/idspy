from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from ......core.step.base import Step
from ......nn.torch.model.base import BaseModel, ModelOutput
from ......nn.torch.loss.base import BaseLoss
from ......nn.torch.engine.loops import eval_epoch
from ......nn.torch.engine.forward import forward_pass, make_predictions
from .... import StepFactory


@StepFactory.register()
@Step.needs("dataloader", "model", "loss_fn", "device")
class ValidateOneEpoch(Step):
    """Validate a model for one epoch (no gradient updates)."""

    def __init__(
        self,
        save_outputs: bool = False,
        dataloader_key: str = "val.dataloader",
        model_key: str = "model",
        loss_key: str = "loss_fn",
        device_key: str = "device",
        metrics_key: str = "val.metrics",
        outputs_key: str = "val.outputs",
        name: Optional[str] = None,
    ) -> None:
        self.save_outputs = save_outputs
        super().__init__(name=name or "validate_one_epoch")

        self.key_map = {
            "dataloader": dataloader_key,
            "model": model_key,
            "loss_fn": loss_key,
            "device": device_key,
            "metrics": metrics_key,
            "outputs": outputs_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: BaseModel,
        device: torch.device,
        loss_fn: Optional[BaseLoss] = None,
        context: Optional[any] = None,
    ) -> Optional[Dict[str, Any]]:
        average_loss, model_outputs = eval_epoch(
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss_fn,
            save_outputs=self.save_outputs,
            profiler=context,
        )

        return {
            "model": model,
            "metrics": {"avg_loss": average_loss},
            "outputs": model_outputs,
        }


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
@Step.needs("inputs", "pred_fn")
class MakePredictions(Step):
    """Make predictions from model outputs."""

    def __init__(
        self,
        inputs_key: str = "inputs",
        pred_key: str = "pred_fn",
        outputs_key: str = "outputs",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "make_predictions")
        self.key_map = {
            "inputs": inputs_key,
            "outputs": outputs_key,
            "pred_fn": pred_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, inputs: torch.Tensor, pred_fn: Callable
    ) -> Optional[Dict[str, Any]]:
        predictions = make_predictions(inputs, pred_fn).detach().cpu().numpy()
        return {"outputs": predictions}
