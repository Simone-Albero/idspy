from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from ....core.step.base import Step
from ....core.step.fittable import FittableStep


@Step.needs("df")
class StandardScale(FittableStep):
    """Standardize numerical columns using mean/std with overflow-safe scaling."""

    def __init__(
        self,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        self._scale: Optional[pd.Series] = None
        self._means_s: Optional[pd.Series] = None
        self._stds_s: Optional[pd.Series] = None

        super().__init__(name=name or "standard_scale")
        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def fit_impl(self, df: pd.DataFrame) -> None:
        """Fit scaling stats on train split (overflow-safe)."""

        numerical_data = df.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._means_s = pd.Series(dtype="float64")
            self._stds_s = pd.Series(dtype="float64")
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

        # Overflow-safe: compute scale and scaled values efficiently
        abs_max = numerical_data.abs().max(axis=0)
        self._scale = (
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
        )

        num_scaled = numerical_data / self._scale
        self._means_s = num_scaled.mean()
        self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)

    def run(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply standardization to numerical columns."""

        numerical_data = df.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"df": df}

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

        cols = numerical_data.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._means_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)

        df.tab.numerical = (numerical_data / scale - means_s) / stds_s
        return {"df": df}


@Step.needs("df")
class MinMaxScale(FittableStep):
    """Scale numerical columns to [0, 1] via min/max."""

    def __init__(
        self,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        self._min: Optional[pd.Series] = None
        self._max: Optional[pd.Series] = None

        super().__init__(name=name or "min_max_scale")
        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def fit_impl(self, df: pd.DataFrame) -> None:
        """Fit min/max on train split."""
        numerical_data = df.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._min = pd.Series(dtype="float64")
            self._max = pd.Series(dtype="float64")
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )
        self._min = numerical_data.min()
        self._max = numerical_data.max()

    def run(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply min-max scaling to numerical columns."""

        numerical_data = df.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"df": df}

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

        cols = numerical_data.columns
        col_min = self._min.reindex(cols, fill_value=0.0)
        col_max = self._max.reindex(cols, fill_value=1.0)
        den = (col_max - col_min).clip(lower=1e-10, upper=None)

        df.tab.numerical = (numerical_data - col_min) / den
        return {"df": df}
