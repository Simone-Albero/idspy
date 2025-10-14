from typing import Any, Dict, Optional

import pandas as pd

from ....core.step.base import Step
from ....data.partition import random_split, stratified_split
from ..factory import StepFactory


@StepFactory.register()
@Step.needs("df", "seed")
class RandomSplit(Step):
    """Random split into train/val/test."""

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        df_key: str = "data.base_df",
        seed_key: str = "seed",
        split_mapping_key: str = "data.split_mapping",
        name: Optional[str] = None,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        super().__init__(name=name or "random_split")
        self.key_map = {
            "df": df_key,
            "seed": seed_key,
            "split_mapping": split_mapping_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame, seed: int) -> Optional[Dict[str, Any]]:

        if df.empty:
            return {"split_mapping": {}, "df_out": df}

        split_mapping = random_split(
            df,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=seed,
        )

        df.tab.set_partitions_from_labels(split_mapping)
        return {"split_mapping": split_mapping, "df": df}


@StepFactory.register()
@Step.needs("df", "seed")
class StratifiedSplit(Step):
    """Stratified split into train/val/test."""

    def __init__(
        self,
        class_column: str,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        df_key: str = "data.base_df",
        seed_key: str = "seed",
        split_mapping_key: str = "data.split_mapping",
        name: Optional[str] = None,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.class_column = class_column

        super().__init__(name=name or "stratified_split")
        self.key_map = {
            "df": df_key,
            "seed": seed_key,
            "split_mapping": split_mapping_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame, seed: int) -> Optional[Dict[str, Any]]:

        if not isinstance(self.class_column, str):
            raise ValueError("stratified_split: 'class_column' must be a string.")

        if df.empty:
            return {"split_mapping": {}, "df_out": df}

        split_mapping = stratified_split(
            df,
            self.class_column,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=seed,
        )

        df.tab.set_partitions_from_labels(split_mapping)
        return {"split_mapping": split_mapping, "df": df}


@StepFactory.register()
@Step.needs("df")
class AllocateSplitPartitions(Step):
    def __init__(
        self,
        df_key: str = "data.base_df",
        train_key: str = "train.data",
        val_key: str = "val.data",
        test_key: str = "test.data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "allocate_split_partitions")
        self.key_map = {
            "df": df_key,
            "train": train_key,
            "val": val_key,
            "test": test_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        return {"train": df.tab.train, "val": df.tab.val, "test": df.tab.test}


@StepFactory.register()
@Step.needs("df")
class AllocateTargets(Step):
    def __init__(
        self,
        df_key: str = "test.data",
        targets_key: str = "test.targets",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "allocate_targets")
        self.key_map = {
            "df": df_key,
            "targets": targets_key,
        }

    def bindings(self) -> Dict[str, str]:
        return self.key_map

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        return {"targets": df.tab.target.to_numpy()}
