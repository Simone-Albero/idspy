from typing import Optional, Any, Dict

import pandas as pd

from ......core.step.base import Step
from ......data.torch.dataset import (
    CategoricalTensorDataset,
    NumericalTensorDataset,
    MixedTabularDataset,
)
from .... import StepFactory


@StepFactory.register()
@Step.needs("df")
class BuildDataset(Step):
    """Build dataset from dataframe in state."""

    def __init__(
        self,
        df_key: str = "data",
        dataset_key: str = "dataset",
        target_col: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name or "build_dataset")
        self.target_col = target_col
        self.key_map = {
            "df": df_key,
            "dataset": dataset_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        numerical_cols = df.tab.schema.numerical
        categorical_cols = df.tab.schema.categorical

        if numerical_cols and categorical_cols:
            dataset = MixedTabularDataset(
                df,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                target_col=self.target_col,
            )
        elif numerical_cols:
            dataset = NumericalTensorDataset(
                df,
                feature_cols=numerical_cols,
                target_col=self.target_col,
            )
        elif categorical_cols:
            dataset = CategoricalTensorDataset(
                df,
                feature_cols=categorical_cols,
                target_col=self.target_col,
            )
        else:
            raise ValueError(
                f"{self.name}: no numerical or categorical columns defined in schema."
            )

        return {"dataset": dataset}
