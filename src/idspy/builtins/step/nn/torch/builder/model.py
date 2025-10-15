from typing import Optional, Any, Dict

from idspy.nn import torch

from ......core.step.base import Step

from .... import StepFactory
from ......nn.torch.model import ModelFactory


@StepFactory.register()
@Step.needs("device")
class BuildModel(Step):
    """build a BaseModel instance from config and Model name."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        model_key: str = "model",
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_model")
        self.model_config = model_config
        self.key_map = {
            "model": model_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, device: torch.device) -> Optional[Dict[str, Any]]:
        model = ModelFactory.create(self.model_config).to(device)
        return {"model": model}
