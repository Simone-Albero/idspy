from pathlib import Path
from typing import Optional, Any, Dict, Union, List

import pandas as pd

from ....core.step.base import Step
from ....data.repository import DataFrameRepository
from ....data.schema import ColumnRole, Schema
from .. import StepFactory


@StepFactory.register()
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

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        DataFrameRepository.save(
            df,
            self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            save_meta=self.save_meta,
            **self.kwargs,
        )


@StepFactory.register()
class LoadData(Step):
    """Load data into state."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fmt: Optional[str] = None,
        file_name: Optional[str] = None,
        load_meta: bool = True,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.fmt = fmt
        self.file_name = file_name
        self.load_meta = load_meta
        self.numerical_cols = list(numerical_cols) if numerical_cols else None
        self.categorical_cols = list(categorical_cols) if categorical_cols else None
        self.label_col = label_col
        self.kwargs = kwargs

        super().__init__(name=name or "load_data")
        self.key_map = {"df": df_key}

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self) -> Optional[Dict[str, Any]]:
        schema = None
        if (
            self.numerical_cols is not None
            or self.categorical_cols is not None
            or self.label_col is not None
        ):
            schema = Schema(
                {
                    ColumnRole.NUMERICAL: self.numerical_cols or [],
                    ColumnRole.CATEGORICAL: self.categorical_cols or [],
                    ColumnRole.LABEL: [self.label_col] if self.label_col else [],
                }
            )

        dataframe: pd.DataFrame = DataFrameRepository.load(
            base_path=self.file_path,
            name=self.file_name,
            fmt=self.fmt,
            schema=schema,
            load_meta=self.load_meta,
            **self.kwargs,
        )
        return {"df": dataframe}
