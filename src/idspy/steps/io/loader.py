from pathlib import Path
from typing import Optional, Any, Dict, Union

import pandas as pd
import torch

from ...core.step.base import Step
from ...data.repository import DataFrameRepository
from ...data.schema import Schema
from ...nn.models.base import BaseModel
from ...nn.io import load_weights


class LoadData(Step):
    """Load data into state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        load_meta: bool = True,
        schema: Optional[Schema] = None,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.load_meta = load_meta
        self.schema = schema
        self.kwargs = kwargs

        super().__init__(name=name or "load_data")
        self.key_map = {"df": df_key}

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self) -> Optional[Dict[str, Any]]:
        dataframe: pd.DataFrame = DataFrameRepository.load(
            base_path=self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            schema=self.schema,
            load_meta=self.load_meta,
            **self.kwargs,
        )
        return {"df": dataframe}


@Step.needs("model", "device")
class LoadModelWeights(Step):
    """Load model from state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        model_key: str = "model",
        device_key: str = "device",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.kwargs = kwargs

        super().__init__(name=name or "load_model_weights")
        self.key_map = {
            "model": model_key,
            "device": device_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, model: BaseModel, device: torch.device) -> Optional[Dict[str, Any]]:
        load_weights(
            model,
            self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            map_location=device,
            **self.kwargs,
        )
        return {"model": model}
