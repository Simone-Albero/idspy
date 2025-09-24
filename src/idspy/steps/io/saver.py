from pathlib import Path
from typing import Optional, Any, Dict, Union

import pandas as pd

from ...core.step import Step
from ...core.state import State
from ...data.repository import DataFrameRepository
from ...nn.models.base import BaseModel
from ...nn.io import save_weights


class SaveData(Step):
    """Save data from state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        save_meta: bool = True,
        in_scope: str = "data",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.save_meta = save_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "save_data",
            in_scope=in_scope,
        )

    @Step.requires(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        DataFrameRepository.save(
            root,
            self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            save_meta=self.save_meta,
            **self.kwargs,
        )


class SaveModelWeights(Step):
    """Save model from state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        in_scope: str = "",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.kwargs = kwargs

        super().__init__(
            name=name or "save_model_weights",
            in_scope=in_scope,
        )

    @Step.requires(model=BaseModel)
    def run(self, state: State, model: BaseModel) -> None:
        save_weights(
            model, self.file_path, name=self.file_name, fmt=self.fmt, **self.kwargs
        )
