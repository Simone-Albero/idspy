from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


from ...core.step.base import Step
from ...core.step.fittable import FittableStep


@Step.needs("df")
class FrequencyMap(FittableStep):
    """Map categorical columns to frequency-rank codes."""

    def __init__(
        self,
        max_levels: Optional[int] = None,
        default: int = 0,
        df_key: str = "data.base_df",
        cat_mapping_key: str = "data.cat_mapping",
        name: Optional[str] = None,
    ) -> None:
        self.max_levels = max_levels
        self.default = default
        self.cat_types: Dict[str, CategoricalDtype] = {}

        super().__init__(name=name or "frequency_map")
        self.key_map = {
            "df": df_key,
            "cat_mapping": cat_mapping_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def fit_impl(self, df: pd.DataFrame) -> None:
        """Infer ordered categories by frequency from train split."""
        train_df = df.tab.train
        self.cat_types.clear()

        cat_cols = train_df.tab.categorical.columns
        for col in cat_cols:
            vc = train_df[col].value_counts(dropna=False)
            if vc.empty:
                continue

            if self.max_levels is None:
                cats = vc.index.tolist()
            else:
                cats = vc.head(self.max_levels).index.tolist()

            self.cat_types[col] = CategoricalDtype(categories=cats, ordered=True)

    def run(self, df: pd.DataFrame) -> None:
        """Apply learned frequency mapping to categorical columns."""

        # Early exit if no categorical mappings learned
        if not self.cat_types:
            return {"df": df, "cat_mapping": {}}

        cat_cols = df.tab.categorical.columns
        for col in cat_cols:
            if col not in self.cat_types or col not in df.columns:
                continue

            s = df[col].astype(self.cat_types[col])
            codes = s.cat.codes
            df[col] = np.where(codes != -1, codes + 1, self.default).astype("int32")

        return {"df": df, "cat_mapping": self.cat_types}


@Step.needs("df")
class LabelMap(FittableStep):
    """Encode `target`: binary with `benign_tag`, else ordinal categories."""

    def __init__(
        self,
        benign_tag: Optional[str] = None,
        default: int = -1,
        df_key: str = "data.base_df",
        target_mapping_key: str = "data.target_mapping",
        name: Optional[str] = None,
    ) -> None:
        self.benign_tag = benign_tag
        self.default = default
        self.cat_types: Optional[CategoricalDtype] = None

        super().__init__(name=name or "label_map")
        self.key_map = {
            "df": df_key,
            "target_mapping": target_mapping_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def fit_impl(self, df: pd.DataFrame) -> None:
        """Learn ordered categories for the target col (if not binary)."""

        # Early exit for binary case
        if self.benign_tag is not None:
            self.cat_types = None
            return

        train_df = df.tab.train
        tgt_col = train_df.tab.schema.target

        vc = train_df[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)

    def run(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        tgt_col = df.tab.schema.target

        prev = df[tgt_col].copy()

        if self.benign_tag is not None:
            tgt = (prev == self.benign_tag).astype("int32")
            tgt = tgt.where(tgt == 0, 1)
        else:
            s = prev.astype(self.cat_types)
            codes = s.cat.codes
            tgt = pd.Series(
                np.where(codes != -1, codes, self.default).astype("int32"),
                index=s.index,
                name=tgt_col,
            )

        df[f"original_{tgt_col}"] = prev
        df.tab.target = tgt
        return {"df": df, "target_mapping": self.cat_types}
