from typing import Optional, Any, Dict, List

import torch

from ......nn.torch.model.base import ModelOutput
from ......core.step.base import Step
from ....factory import StepFactory


@StepFactory.register()
@Step.needs("inputs")
class CatTensors(Step):
    """Stack model outputs into a single tensor."""

    def __init__(
        self,
        inputs_key: str = "inputs",
        input_section: str = "logits",
        outputs_key: str = "outputs",
        to_cpu: bool = False,
        to_numpy: bool = False,
        cat_dim: int = 0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "cat_tensors")

        self.cat_dim = cat_dim
        self.input_section = input_section
        self.to_cpu = to_cpu
        self.to_numpy = to_numpy
        self.key_map = {
            "outputs": outputs_key,
            "inputs": inputs_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, inputs: List[ModelOutput]) -> Optional[Dict[str, Any]]:
        outputs = []
        for input in inputs:
            outputs.append(getattr(input, self.input_section))

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=self.cat_dim)
            if self.to_cpu:
                outputs = outputs.to(torch.device("cpu"))
            if self.to_numpy:
                outputs = outputs.numpy()

        return {"outputs": outputs}
