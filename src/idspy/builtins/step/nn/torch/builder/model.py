from typing import Optional, Any, Dict

import torch

from ......core.step.base import Step

from .... import StepFactory
from ......nn.torch.model import ModelFactory


@StepFactory.register()
@Step.needs("device")
class BuildModel(Step):
    """build a BaseModel instance from config and Model name."""

    def __init__(
        self,
        model_name: str,
        model_args: Dict[str, Any],
        model_key: str = "model",
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_model")
        self.model_name = model_name
        self.model_args = model_args
        self.key_map = {
            "model": model_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, device: torch.device) -> Optional[Dict[str, Any]]:
        model = ModelFactory.create(self.model_name, self.model_args).to(device)
        return {"model": model}
