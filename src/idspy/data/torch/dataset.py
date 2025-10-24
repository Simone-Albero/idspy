from typing import Mapping, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset


Sample = Mapping[str, torch.Tensor]


class TensorDataset(Dataset):
    """
    Wraps a pandas DataFrame (and optional pandas Series) into torch tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        feature_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ) -> None:
        self.features: torch.Tensor = torch.as_tensor(df.values, dtype=feature_dtype)
        self.labels: Optional[torch.Tensor] = (
            torch.as_tensor(labels.values, dtype=label_dtype)
            if labels is not None
            else None
        )

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> Sample:
        features = self.features[index]
        labels = self.labels[index] if self.labels is not None else features
        return {"features": features, "targets": labels}


class NumericalTensorDataset(TensorDataset):
    """
    Dataset wrapper for numerical features in a DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        label_col: Optional[str] = None,
    ) -> None:
        df = df[feature_cols]
        labels = df[label_col] if label_col else None
        super().__init__(
            df,
            labels,
            feature_dtype=torch.float32,
            label_dtype=torch.long,
        )


class CategoricalTensorDataset(TensorDataset):
    """
    Dataset wrapper for categorical features in a DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        label_col: Optional[str] = None,
    ) -> None:
        df = df[feature_cols]
        labels = df[label_col] if label_col else None

        super().__init__(
            df,
            labels,
            feature_dtype=torch.long,
            label_dtype=torch.long,
        )


class MixedTabularDataset(Dataset):
    """
    Combines numerical and categorical datasets into a single dataset yielding TabularSample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numerical_cols: Sequence[str],
        categorical_cols: Sequence[str],
        label_col: Optional[str] = None,
    ) -> None:
        self.numerical_ds = NumericalTensorDataset(df, numerical_cols, label_col=None)
        self.categorical_ds = CategoricalTensorDataset(
            df, categorical_cols, label_col=None
        )

        self.labels: Optional[torch.Tensor] = (
            torch.as_tensor(df[label_col].values, dtype=torch.long)
            if label_col
            else None
        )

    def __len__(self) -> int:
        return len(self.numerical_ds)

    def __getitem__(self, index: int) -> Sample:
        numerical_sample = self.numerical_ds[index]
        categorical_sample = self.categorical_ds[index]
        features = {
            "numerical": numerical_sample["features"],
            "categorical": categorical_sample["features"],
        }
        labels = self.labels[index] if self.labels is not None else features
        return {"features": features, "targets": labels}
