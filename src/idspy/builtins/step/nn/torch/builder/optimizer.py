from typing import Optional, Any, Dict

import torch

from ......core.step.base import Step

from .... import StepFactory


@StepFactory.register()
@Step.needs("model", "loss_fn")
class BuildOptimizer(Step):
    """build an optimizer instance from config and model."""

    def __init__(
        self,
        optimizer_args: Dict[str, Any],
        model_key: str = "model",
        optimizer_key: str = "optimizer",
        loss_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_model")
        self.optimizer_args = optimizer_args
        self.key_map = {
            "model": model_key,
            "optimizer": optimizer_key,
        }
        if loss_key is not None:
            self.key_map["loss_fn"] = loss_key

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, model: torch.nn.Module, loss_fn: Optional[torch.nn.Module] = None
    ) -> Optional[Dict[str, Any]]:
        OptimizerClass = getattr(torch.optim, self.optimizer_args._target_)

        params = list(model.parameters())
        if loss_fn is not None:
            loss_params = list(loss_fn.parameters())
            if loss_params:
                params.extend(loss_params)

        optimizer = OptimizerClass(params, **self.optimizer_args._params_)
        return {"optimizer": optimizer}
