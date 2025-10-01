from typing import Dict, Optional, Any

import torch

from .....common.utils import PathLike
from .....core.step.base import Step
from .....nn.torch.model.base import BaseModel
from .....nn.torch.module.repository import ModulesRepository


@Step.needs("model", "device")
class LoadModelWeights(Step):
    """Load model from state."""

    def __init__(
        self,
        file_path: PathLike,
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        model_key: str = "model",
        device_key: str = "device",
        name: Optional[str] = None,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name

        super().__init__(name=name or "load_model_weights")
        self.key_map = {
            "model": model_key,
            "device": device_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, model: BaseModel, device: torch.device) -> Optional[Dict[str, Any]]:
        ModulesRepository.load_weights(
            model,
            self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            map_location=device,
        )
        return {"model": model}


@Step.needs("model")
class SaveModelWeights(Step):
    """Save model from state."""

    def __init__(
        self,
        file_path: PathLike,
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        model_key: str = "model",
        name: Optional[str] = None,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name

        super().__init__(name=name or "save_model_weights")
        self.key_map = {"model": model_key}

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, model: BaseModel) -> Optional[Dict[str, Any]]:
        ModulesRepository.save_weights(
            model, self.file_path, name=self.file_name, fmt=self.fmt
        )
