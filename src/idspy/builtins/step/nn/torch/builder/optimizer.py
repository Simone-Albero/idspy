from typing import Optional, Any, Dict

import torch

from ......core.step.base import Step

from .... import StepFactory


@StepFactory.register()
@Step.needs("model")
class BuildOptimizer(Step):
    """build an optimizer instance from config and model."""

    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        model_key: str = "model",
        optimizer_key: str = "optimizer",
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_model")
        self.optimizer_config = optimizer_config
        self.key_map = {
            "model": model_key,
            "optimizer": optimizer_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, model: torch.nn.Module) -> Optional[Dict[str, Any]]:
        OptimizerClass = getattr(torch.optim, self.optimizer_config._target_)
        optimizer = OptimizerClass(model.parameters(), **self.optimizer_config._params_)
        return {"optimizer": optimizer}
