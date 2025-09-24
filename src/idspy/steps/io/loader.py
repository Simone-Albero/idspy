from pathlib import Path
from sre_parse import State
from typing import Optional, Any, Dict, Union

import pandas as pd
import torch

from ...core.step import Step
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
        schema: Optional[Schema] = None,
        load_meta: bool = True,
        in_scope: Optional[str] = "data",
        out_scope: Optional[str] = "data",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.schema = schema
        self.load_meta = load_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "load_data",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.provides(root=pd.DataFrame)
    def run(self, state: State) -> Optional[Dict[str, Any]]:
        dataframe: pd.DataFrame = DataFrameRepository.load(
            base_path=self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            schema=self.schema,
            load_meta=self.load_meta,
            **self.kwargs,
        )
        return {"root": dataframe}


class LoadModelWeights(Step):
    """Load model from state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
        out_scope: str = "",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.map_location = map_location
        self.kwargs = kwargs

        super().__init__(
            name=name or "load_model_weights",
            out_scope=out_scope,
        )

    @Step.requires(model=BaseModel)
    def run(self, state: State, model: BaseModel) -> None:
        load_weights(
            model,
            self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            map_location=self.map_location,
            **self.kwargs,
        )
