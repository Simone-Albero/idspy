from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ....core.step.base import Step
from ....data.tab_accessor import reattach_meta
from .. import StepFactory


@StepFactory.register()
@Step.needs("df")
class DropNulls(Step):
    """Drop all rows that contain null values, including NaN and ±inf."""

    def __init__(
        self,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "drop_nulls")

        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return {"df": df}


@StepFactory.register()
@Step.needs("df")
class Filter(Step):
    """Filter rows using a pandas query string."""

    def __init__(
        self,
        query: str,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        self.query = query

        super().__init__(name=name or "filter")

        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        filtered = df.query(self.query)
        return {"df": reattach_meta(df, filtered)}


@StepFactory.register()
@Step.needs("df")
class RareClassFilter(Step):
    """Filter out rare classes from a categorical column."""

    def __init__(
        self,
        target_col: str,
        min_count: int = 100,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        self.target_col = target_col
        self.min_count = min_count

        super().__init__(name=name or "rare_class_filter")

        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        value_counts = df[self.target_col].value_counts()
        to_keep = value_counts[value_counts >= self.min_count].index
        filtered = df[df[self.target_col].isin(to_keep)]
        return {"df": reattach_meta(df, filtered)}


@StepFactory.register()
@Step.needs("df")
class Log1p(Step):
    """Apply np.log1p to numerical columns."""

    def __init__(
        self,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "log1p")

        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        df.tab.numerical = np.log1p(df.tab.numerical)
        return {"df": df}


@StepFactory.register()
@Step.needs("df")
class ToNumpy(Step):
    def __init__(
        self,
        df_key: str = "test.data",
        output_key: str = "test.targets",
        columns: Optional[list] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "to_numpy")
        self.columns = columns
        self.key_map = {
            "df": df_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if self.columns is not None:
            # Convert OmegaConf to native Python and resolve interpolations
            columns = OmegaConf.to_container(self.columns, resolve=True)

            # Flatten nested lists recursively
            def flatten(lst):
                result = []
                for item in lst:
                    if isinstance(item, list):
                        result.extend(flatten(item))
                    else:
                        result.append(item)
                return result

            columns = flatten(columns) if len(columns) > 1 else str(columns[0])
            return {"output": df[columns].values}

        return {"output": df.values}
