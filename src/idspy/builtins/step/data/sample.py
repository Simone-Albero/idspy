from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from ....core.step.base import Step
from ....data.tab_accessor import reattach_meta
from ..helpers import sample_vectors_and_labels
from .. import StepFactory


@StepFactory.register()
@Step.needs("df")
class DownsampleToMinority(Step):
    """Downsample each class to the size of the minority class."""

    def __init__(
        self,
        class_col: str,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        self.class_col = class_col

        super().__init__(name=name or "downsample_to_minority")
        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:

        # Early exits for edge cases
        if df.empty or self.class_col not in df.columns:
            return {"df": df}

        counts = df[self.class_col].value_counts(dropna=False)
        if counts.empty:
            return {"df": df}

        minority = int(counts.min())
        if minority <= 0:
            sampled = df.iloc[0:0]  # empty but keep schema
            return {"df": reattach_meta(df, sampled)}

        sampled = df.groupby(
            self.class_col, dropna=False, group_keys=False, sort=False
        ).sample(n=minority, replace=False, random_state=self.random_state)

        return {"df": reattach_meta(df, sampled)}


@StepFactory.register()
@Step.needs("df")
class Downsample(Step):
    """Downsample rows globally or per class."""

    def __init__(
        self,
        frac: float,
        class_col: Optional[str] = None,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"downsample: frac must be in (0, 1], got {frac}.")

        self.frac = frac
        self.class_col = class_col

        super().__init__(name=name or "downsample")
        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:

        if df.empty:
            return {"df": df}

        if self.class_col is not None and self.class_col in df.columns:
            sampled = df.groupby(
                self.class_col, dropna=False, group_keys=False, sort=False
            ).sample(frac=self.frac, replace=False, random_state=self.random_state)
        else:
            # Global sampling (handles both None class_column and missing column cases)
            sampled = df.sample(
                frac=self.frac, replace=False, random_state=self.random_state
            )

        return {"df": reattach_meta(df, sampled)}


@StepFactory.register()
@Step.needs("vectors", "labels")
class SampleVectorsAndLabels(Step):

    def __init__(
        self,
        sample_size: int,
        stratify: bool = True,
        random_state: Optional[int] = None,
        vectors_key: str = "data.base_df",
        labels_key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.sample_size = sample_size
        self.stratify = stratify
        self.random_state = random_state

        super().__init__(name=name or "sample_vectors_and_labels")
        self.key_map = {
            "vectors": vectors_key,
        }
        if labels_key is not None:
            self.key_map["labels"] = labels_key

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, vectors: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:

        vectors, labels = sample_vectors_and_labels(
            vectors, labels, self.sample_size, self.stratify, self.random_state
        )

        return {
            "vectors": vectors,
            "labels": labels,
        }
