from pathlib import Path
from typing import Optional, Any, Dict, Union

import pandas as pd

from ...core.step.base import Step
from ...data.repository import DataFrameRepository
from ...nn.models.base import BaseModel
from ...nn.io import save_weights


@Step.needs("df")
class SaveData(Step):
    """Save data from state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        save_meta: bool = True,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.save_meta = save_meta
        self.kwargs = kwargs

        super().__init__(name=name or "save_data")
        self.key_map = {"df": df_key}

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        DataFrameRepository.save(
            df,
            self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            save_meta=self.save_meta,
            **self.kwargs,
        )


@Step.needs("model")
class SaveModelWeights(Step):
    """Save model from state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        model_key: str = "model",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.kwargs = kwargs

        super().__init__(name=name or "save_model_weights")
        self.key_map = {"model": model_key}

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, model: BaseModel) -> Optional[Dict[str, Any]]:
        save_weights(
            model, self.file_path, name=self.file_name, fmt=self.fmt, **self.kwargs
        )
