from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from ....core.step.base import Step
from ....data.tab_accessor import reattach_meta
from ..helpers import sample_vectors_and_labels, sample_labels
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
        n_samples: int,
        class_col: Optional[str] = None,
        fair: bool = False,
        df_key: str = "data.base_df",
        random_state: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if n_samples <= 0:
            raise ValueError(
                f"downsample: n_samples must be positive, got {n_samples}."
            )

        self.n_samples = n_samples
        self.class_col = class_col
        self.fair = fair
        self.random_state = random_state

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
            if self.fair:
                # Fair sampling: equal samples per class
                counts = df[self.class_col].value_counts(dropna=False)
                n_classes = len(counts)
                samples_per_class = self.n_samples // n_classes

                if samples_per_class <= 0:
                    sampled = df.iloc[0:0]  # empty but keep schema
                    return {"df": reattach_meta(df, sampled)}

                # Sample each class, taking min of requested and available
                sampled = df.groupby(
                    self.class_col, dropna=False, group_keys=False, sort=False
                ).sample(
                    n=lambda x: min(len(x), samples_per_class),
                    replace=False,
                    random_state=self.random_state,
                )
            else:
                # Proportional sampling: maintain class distribution
                frac = self.n_samples / len(df)
                if frac > 1.0:
                    frac = 1.0

                sampled = df.groupby(
                    self.class_col, dropna=False, group_keys=False, sort=False
                ).sample(frac=frac, replace=False, random_state=self.random_state)
        else:
            # Global sampling (handles both None class_column and missing column cases)
            n = min(self.n_samples, len(df))
            sampled = df.sample(n=n, replace=False, random_state=self.random_state)

        return {"df": reattach_meta(df, sampled)}


@StepFactory.register()
@Step.needs("labels")
class ComputeIndicesByLabel(Step):
    """Select sample indices from labels with optional stratification."""

    def __init__(
        self,
        sample_size: int,
        stratify: bool = True,
        random_state: Optional[int] = None,
        labels_key: str = "data.labels",
        indices_key: str = "data.selected_indices",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "compute_indices_by_label")
        self.sample_size = sample_size
        self.stratify = stratify
        self.random_state = random_state
        self.key_map = {
            "labels": labels_key,
            "selected_indices": indices_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, labels: np.ndarray) -> Optional[Dict[str, Any]]:
        samples = sample_labels(
            labels, self.sample_size, self.stratify, self.random_state
        )
        return {"selected_indices": samples}


@StepFactory.register()
@Step.needs("data", "selected_indices")
class SelectSamplesByIndices(Step):
    """Select samples from data based on provided indices."""

    def __init__(
        self,
        indices_key: str = "data.selected_indices",
        data_key: str = "data.base_df",
        output_key: str = "data.sampled_df",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "select_samples_by_indices")
        self.key_map = {
            "selected_indices": indices_key,
            "data": data_key,
            "sampled_data": output_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(
        self, data: pd.DataFrame, selected_indices: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        sampled_data = data[selected_indices]
        return {"sampled_data": sampled_data}
