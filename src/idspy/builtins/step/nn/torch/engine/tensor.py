from typing import Optional, Any, Dict, List

import torch
import numpy as np

from ......nn.torch.model.base import ModelOutput
from ......core.step.base import Step
from .... import StepFactory


@StepFactory.register()
@Step.needs("tensors")
class CatTensors(Step):
    """Concatenate a list of tensors along a specified dimension."""

    def __init__(
        self,
        tensors_key: str = "tensors",
        output_key: str = "output",
        to_cpu: bool = False,
        to_numpy: bool = False,
        cat_dim: int = 0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "cat_tensors")

        self.cat_dim = cat_dim
        self.to_cpu = to_cpu
        self.to_numpy = to_numpy
        self.key_map = {
            "output": output_key,
            "tensors": tensors_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, tensors: List[torch.Tensor]) -> Optional[Dict[str, Any]]:
        output = torch.cat(tensors, dim=self.cat_dim)

        if self.to_cpu:
            output = output.to(torch.device("cpu"))
        if self.to_numpy:
            output = output.numpy()

        return {"output": output}


@StepFactory.register()
@Step.needs("array")
class ToTensor(Step):
    def __init__(
        self,
        array_key: str = "test.features",
        output_key: str = "test.features_tensor",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "to_tensor")
        self.key_map = {
            "array": array_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, array: np.ndarray) -> Optional[Dict[str, Any]]:
        tensor = torch.from_numpy(array)
        return {"output": tensor}


@StepFactory.register()
@Step.needs("tensor")
class ToArray(Step):
    def __init__(
        self,
        tensor_key: str = "tensor",
        output_key: str = "array",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "to_array")
        self.key_map = {
            "tensor": tensor_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        array = tensor.detach().cpu().numpy()
        return {"output": array}
