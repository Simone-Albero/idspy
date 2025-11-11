from typing import Any, Dict, Optional, List, Union

import numpy as np
import pandas as pd

from ....core.step.base import Step
from ....core.step.fittable import FittableStep
from ....core.step.conditional import ConditionalStep
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
class Filter(ConditionalStep):
    """Filter rows using a pandas query string."""

    def __init__(
        self,
        df_key: str = "data.base_df",
        query: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.query = query

        super().__init__(name=name or "filter")

        self.key_map = {
            "df": df_key,
        }

    def should_run(self, **kwargs):
        return self.query is not None

    def on_skip(self, **kwargs):
        return

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
class DFToNumpy(Step):
    """Convert specified columns of a DataFrame to a NumPy array."""

    def __init__(
        self,
        df_key: str = "test.data",
        output_key: str = "test.labels",
        cols: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "to_numpy")
        self.cols = cols
        self.key_map = {
            "df": df_key,
            "output": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if self.cols is not None:
            if isinstance(self.cols, str):
                return {"output": df[self.cols].values}

            return {"output": df[self.cols].values}

        return {"output": df.values}


@StepFactory.register()
@Step.needs("df")
class Clip(FittableStep):
    """Clip extreme values based on percentiles or absolute thresholds."""

    def __init__(
        self,
        df_key: str = "data.base_df",
        method: str = "percentile",  # "percentile" or "absolute"
        lower: float = 1.0,  # 1st percentile or absolute value
        upper: float = 99.0,  # 99th percentile or absolute value
        name: Optional[str] = None,
    ) -> None:
        self._lower_bounds: Optional[pd.Series] = None
        self._upper_bounds: Optional[pd.Series] = None
        self.method = method
        self.lower = lower
        self.upper = upper

        super().__init__(name=name or "clip_outliers")
        self.key_map = {"df": df_key}

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def fit_impl(self, df: pd.DataFrame) -> None:
        """Fit clipping bounds on train split."""
        numerical_data = df.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._lower_bounds = pd.Series(dtype="float64")
            self._upper_bounds = pd.Series(dtype="float64")
            return

        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)

        if self.method == "percentile":
            self._lower_bounds = numerical_data.quantile(self.lower / 100)
            self._upper_bounds = numerical_data.quantile(self.upper / 100)
        else:  # absolute
            self._lower_bounds = pd.Series(self.lower, index=numerical_data.columns)
            self._upper_bounds = pd.Series(self.upper, index=numerical_data.columns)

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply clipping to numerical columns."""
        numerical_data = df.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"df": df}

        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)

        cols = numerical_data.columns
        lower = self._lower_bounds.reindex(cols, fill_value=-np.inf)
        upper = self._upper_bounds.reindex(cols, fill_value=np.inf)

        df.tab.numerical = numerical_data.clip(lower=lower, upper=upper, axis=1)
        return {"df": df}
