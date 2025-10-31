import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


from ....core.step.base import Step
from ....core.step.fittable import FittableStep
from .. import StepFactory
from ....plot.dict import dict_to_table


logger = logging.getLogger(__name__)


@StepFactory.register()
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
        self.max_levels = (
            max_levels - 1 if max_levels else None
        )  # Reserve 0 for unseen categories
        self.default = default
        self.cat_types: Dict[str, CategoricalDtype] = {}
        self.cat_mapping: Dict[str, Dict[Any, int]] = {}

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

    def compute(self, df: pd.DataFrame) -> None:
        """Apply learned frequency mapping to categorical columns."""

        # Early exit if no categorical mappings learned
        if not self.cat_types:
            return {"df": df, "cat_mapping": self.cat_mapping}

        cat_cols = df.tab.categorical.columns

        for col in cat_cols:
            if col not in self.cat_types or col not in df.columns:
                continue

            s = df[col].astype(self.cat_types[col])
            codes = s.cat.codes
            df[col] = np.where(codes != -1, codes + 1, self.default).astype("int32")

            # Create mapping: category -> integer code (0 for unseen, 1+ for known)
            self.cat_mapping[col] = {
                cat: idx + 1 for idx, cat in enumerate(self.cat_types[col].categories)
            }
            # Add default mapping for unseen categories
            self.cat_mapping[col]["__unseen__"] = self.default

        return {"df": df, "cat_mapping": self.cat_mapping}

    def __del__(self):
        logger.info(f"FrequencyMap mapping:\n{self.cat_mapping}")


@StepFactory.register()
@Step.needs("df")
class LabelMap(FittableStep):
    """Encode `label`: binary with `benign_tag`, else ordinal categories."""

    def __init__(
        self,
        benign_tag: Optional[str] = None,
        default: int = -1,
        df_key: str = "data.base_df",
        target_col: str = "label_encoded",
        labels_mapping_key: str = "data.label_mapping",
        name: Optional[str] = None,
    ) -> None:
        self.benign_tag = benign_tag
        self.default = default
        self.target_col = target_col
        self.cat_types: Optional[CategoricalDtype] = None
        self.labels_mapping: Dict[Any, int] = {}

        super().__init__(name=name or "label_map")
        self.key_map = {
            "df": df_key,
            "labels_mapping": labels_mapping_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def fit_impl(self, df: pd.DataFrame) -> None:
        """Learn ordered categories for the label col (if not binary)."""

        # Early exit for binary case
        if self.benign_tag is not None:
            self.cat_types = None
            return

        train_df = df.tab.train
        tgt_col = train_df.tab.schema.label

        vc = train_df[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        label_col = df.tab.schema.label

        labels = df[label_col].copy()

        if self.benign_tag is not None:
            labels = (labels == self.benign_tag).astype("int32")
            labels = labels.where(labels == 0, 1)

            self.labels_mapping = {self.benign_tag: 0, "others": 1}
        else:
            s = labels.astype(self.cat_types)
            codes = s.cat.codes
            labels = pd.Series(
                np.where(codes != -1, codes, self.default).astype("int32"),
                index=s.index,
                name=label_col,
            )

            self.labels_mapping = {
                cat: idx for idx, cat in enumerate(self.cat_types.categories)
            }

        df[self.target_col] = labels
        df.tab.update_role(cols=self.target_col, role="label")

        return {
            "df": df,
            "labels_mapping": dict_to_table(self.labels_mapping, title="Label Mapping"),
        }

    def __del__(self):
        logger.info(f"LabelMap mapping:\n{self.labels_mapping}")


@StepFactory.register()
@Step.needs("df")
class ColumnMap(FittableStep):
    """Map a single column to frequency-rank codes."""

    def __init__(
        self,
        source_col: str,
        target_col: str,
        max_levels: Optional[int] = None,
        default: int = 0,
        df_key: str = "data.base_df",
        col_mapping_key: str = "data.col_mapping",
        name: Optional[str] = None,
    ) -> None:
        self.source_col = source_col
        self.target_col = target_col
        self.max_levels = max_levels - 1 if max_levels else None  # Reserve 0 for unseen
        self.default = default
        self.cat_type: Optional[CategoricalDtype] = None
        self.col_mapping: Dict[Any, int] = {}

        super().__init__(name=name or f"column_map_{source_col}")
        self.key_map = {
            "df": df_key,
            "col_mapping": col_mapping_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def fit_impl(self, df: pd.DataFrame) -> None:
        """Infer ordered categories by frequency from train split."""
        train_df = df.tab.train

        if self.source_col not in train_df.columns:
            raise KeyError(f"Column '{self.source_col}' not found in dataframe")

        vc = train_df[self.source_col].value_counts(dropna=False)
        if vc.empty:
            self.cat_type = None
            return

        if self.max_levels is None:
            cats = vc.index.tolist()
        else:
            cats = vc.head(self.max_levels).index.tolist()

        self.cat_type = CategoricalDtype(categories=cats, ordered=True)

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply learned frequency mapping to the specified column."""

        if self.cat_type is None or self.source_col not in df.columns:
            return {"df": df, "col_mapping": {}}

        s = df[self.source_col].astype(self.cat_type)
        codes = s.cat.codes
        df[self.target_col] = np.where(codes != -1, codes + 1, self.default).astype(
            "int32"
        )

        # Create mapping: category -> integer code (0 for unseen, 1+ for known)
        self.col_mapping = {
            cat: idx + 1 for idx, cat in enumerate(self.cat_type.categories)
        }
        self.col_mapping["__unseen__"] = self.default

        return {
            "df": df,
            "col_mapping": dict_to_table(
                self.col_mapping, title=f"{self.source_col} Mapping"
            ),
        }

    def __del__(self):
        logger.info(f"ColumnMap mapping for '{self.source_col}':\n{self.col_mapping}")
