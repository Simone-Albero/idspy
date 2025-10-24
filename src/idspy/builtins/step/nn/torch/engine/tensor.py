from typing import Optional, Any, Dict, List

import torch
import pandas as pd
import numpy as np

from ......nn.torch.model.base import ModelOutput
from ......core.step.base import Step
from .... import StepFactory


@StepFactory.register()
@Step.needs("inputs")
class CatTensors(Step):
    """Stack model outputs into a single tensor."""

    def __init__(
        self,
        inputs_key: str = "inputs",
        outputs_key: str = "outputs",
        to_cpu: bool = False,
        to_numpy: bool = False,
        cat_dim: int = 0,
        section: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "cat_tensors")

        self.cat_dim = cat_dim
        self.section = section
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
            outputs.append(input[self.section] if self.section else input)

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=self.cat_dim)
            if self.to_cpu:
                outputs = outputs.to(torch.device("cpu"))
            if self.to_numpy:
                outputs = outputs.numpy()

        return {"outputs": outputs}


@StepFactory.register()
@Step.needs("array")
class ToTensor(Step):
    def __init__(
        self,
        input_key: str = "test.features",
        output_key: str = "test.features_tensor",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "to_tensor")
        self.key_map = {
            "array": input_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, array: np.ndarray) -> Optional[Dict[str, Any]]:
        tensor = torch.from_numpy(array)
        return {"output": tensor}
