from typing import Any, Dict, Optional

import pandas as pd

from ....core.step.base import Step
from ....data.tab_accessor import reattach_meta


@Step.needs("df")
class DownsampleToMinority(Step):
    """Downsample each class to the size of the minority class."""

    def __init__(
        self,
        class_column: str,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        self.class_column = class_column

        super().__init__(name=name or "downsample_to_minority")
        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:

        # Early exits for edge cases
        if df.empty or self.class_column not in df.columns:
            return {"df": df}

        counts = df[self.class_column].value_counts(dropna=False)
        if counts.empty:
            return {"df": df}

        minority = int(counts.min())
        if minority <= 0:
            sampled = df.iloc[0:0]  # empty but keep schema
            return {"df": reattach_meta(df, sampled)}

        sampled = df.groupby(
            self.class_column, dropna=False, group_keys=False, sort=False
        ).sample(n=minority, replace=False, random_state=self.random_state)

        return {"df": reattach_meta(df, sampled)}


@Step.needs("df")
class Downsample(Step):
    """Downsample rows globally or per class."""

    def __init__(
        self,
        frac: float,
        class_column: Optional[str] = None,
        df_key: str = "data.base_df",
        name: Optional[str] = None,
    ) -> None:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"downsample: frac must be in (0, 1], got {frac}.")

        self.frac = frac
        self.class_column = class_column

        super().__init__(name=name or "downsample")
        self.key_map = {
            "df": df_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def run(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:

        if df.empty:
            return {"df": df}

        if self.class_column is not None and self.class_column in df.columns:
            sampled = df.groupby(
                self.class_column, dropna=False, group_keys=False, sort=False
            ).sample(frac=self.frac, replace=False, random_state=self.random_state)
        else:
            # Global sampling (handles both None class_column and missing column cases)
            sampled = df.sample(
                frac=self.frac, replace=False, random_state=self.random_state
            )

        return {"df": reattach_meta(df, sampled)}
