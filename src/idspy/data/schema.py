from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Union, Optional

import pandas as pd


class ColumnRole(Enum):
    """Column roles tracked by the schema."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TARGET = "target"
    FEATURES = "features"

    @classmethod
    def from_name(cls, name: Union[str, "ColumnRole"]) -> "ColumnRole":
        """Coerce a string or enum to ColumnRole."""
        if isinstance(name, ColumnRole):
            return name
        key = str(name).lower()
        for member in cls:
            if member.value == key:
                return member
        raise KeyError(f"Unknown role: {name}")


@dataclass(slots=True, eq=False)
class Schema:
    """Store roles (types) of dataframe columns."""

    @staticmethod
    def _as_list(cols: Union[Iterable[str], str]) -> List[str]:
        """Normalize to list[str]."""
        if isinstance(cols, str):
            return [cols]
        return [c for c in cols]

    roles: Dict[ColumnRole, Union[List[str], str]] = field(
        default_factory=lambda: {
            ColumnRole.NUMERICAL: [],
            ColumnRole.CATEGORICAL: [],
            ColumnRole.TARGET: "",
        }
    )
    strict: bool = False

    def __init__(
        self,
        roles: Optional[Dict[Union[ColumnRole, str], Union[List[str], str]]] = None,
        strict: bool = False,
    ) -> None:
        """Initialize schema with roles dictionary.

        Args:
            roles: Dictionary mapping ColumnRole to columns.
                   TARGET should be a single string or list with one element.
                   NUMERICAL and CATEGORICAL should be lists of strings.
            strict: If True, raise error when pruning missing columns.
        """
        self.roles = {
            ColumnRole.NUMERICAL: [],
            ColumnRole.CATEGORICAL: [],
            ColumnRole.TARGET: "",
        }
        self.strict = strict

        if roles:
            for role, cols in roles.items():
                role = ColumnRole.from_name(role)

                if role == ColumnRole.FEATURES:
                    raise ValueError(
                        "Cannot directly set FEATURES role. It is computed from NUMERICAL + CATEGORICAL."
                    )

                if cols:
                    self.add(cols, role)

    def add(
        self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]
    ) -> None:
        """Add columns to a role, ensuring exclusivity and order."""
        role = ColumnRole.from_name(role)

        if role == ColumnRole.FEATURES:
            raise ValueError(
                "Cannot directly modify FEATURES role. It is computed from NUMERICAL + CATEGORICAL."
            )

        new_cols = self._as_list(cols)

        # Handle TARGET role differently (single string)
        if role == ColumnRole.TARGET:
            if len(new_cols) > 1:
                raise ValueError("TARGET role can only contain one column")
            col_to_add = new_cols[0] if new_cols else ""
            # Remove from other roles
            for r in [ColumnRole.NUMERICAL, ColumnRole.CATEGORICAL]:
                self.roles[r] = [c for c in self.roles[r] if c != col_to_add]
            self.roles[role] = col_to_add
        else:
            # Remove new_cols from TARGET if present
            if self.roles[ColumnRole.TARGET] in new_cols:
                self.roles[ColumnRole.TARGET] = ""

            # Remove from the other list role
            other_role = (
                ColumnRole.CATEGORICAL
                if role == ColumnRole.NUMERICAL
                else ColumnRole.NUMERICAL
            )
            self.roles[other_role] = [
                c for c in self.roles[other_role] if c not in new_cols
            ]

            # Add to the specified role
            seen = set(self.roles[role])
            self.roles[role].extend([c for c in new_cols if c not in seen])

    def update(
        self, cols: Union[Iterable[str], str], role: Union[ColumnRole, str]
    ) -> None:
        """Replace the columns of a role; remove them from other roles."""
        role = ColumnRole.from_name(role)

        if role == ColumnRole.FEATURES:
            raise ValueError(
                "Cannot directly modify FEATURES role. It is computed from NUMERICAL + CATEGORICAL."
            )

        new_cols = self._as_list(cols)

        # Handle TARGET role differently (single string)
        if role == ColumnRole.TARGET:
            if len(new_cols) > 1:
                raise ValueError("TARGET role can only contain one column")
            col_to_set = new_cols[0] if new_cols else ""
            # Remove from other roles
            for r in [ColumnRole.NUMERICAL, ColumnRole.CATEGORICAL]:
                self.roles[r] = [c for c in self.roles[r] if c != col_to_set]
            self.roles[role] = col_to_set
        else:
            # Remove new_cols from TARGET if present
            if self.roles[ColumnRole.TARGET] in new_cols:
                self.roles[ColumnRole.TARGET] = ""

            # Remove from the other list role
            other_role = (
                ColumnRole.CATEGORICAL
                if role == ColumnRole.NUMERICAL
                else ColumnRole.NUMERICAL
            )
            self.roles[other_role] = [
                c for c in self.roles[other_role] if c not in new_cols
            ]

            # Set exact new list (order preserved, dedup)
            seen: set[str] = set()
            self.roles[role] = [c for c in new_cols if not (c in seen or seen.add(c))]

    def columns(self, role: Union[ColumnRole, str]) -> Union[List[str], str]:
        role = ColumnRole.from_name(role)
        if role == ColumnRole.FEATURES:
            return self.features
        return self.roles[role]

    @property
    def numerical(self) -> List[str]:
        return self.roles[ColumnRole.NUMERICAL]

    @property
    def categorical(self) -> List[str]:
        return self.roles[ColumnRole.CATEGORICAL]

    @property
    def target(self) -> str:
        return self.roles[ColumnRole.TARGET]

    @property
    def features(self) -> List[str]:
        """Dynamically compute FEATURES as NUMERICAL + CATEGORICAL, excluding TARGET."""
        target_col = self.roles[ColumnRole.TARGET]
        features_cols = []

        # Preserve order: first numerical, then categorical
        for col in self.roles[ColumnRole.NUMERICAL]:
            if col != target_col:
                features_cols.append(col)

        for col in self.roles[ColumnRole.CATEGORICAL]:
            if col != target_col:
                features_cols.append(col)

        return features_cols

    def prune_missing(self, existing: pd.Index) -> None:
        """Remove columns not present in dataframe."""
        existing_set = set(existing)
        missing: List[str] = []

        for r, cols in self.roles.items():
            if r == ColumnRole.TARGET:
                if cols and cols not in existing_set:
                    missing.append(cols)
                    self.roles[r] = ""
            else:
                keep = [c for c in cols if c in existing_set]
                if len(keep) != len(cols):
                    missing.extend([c for c in cols if c not in existing_set])
                self.roles[r] = keep

        if self.strict and missing:
            raise KeyError(f"Missing columns for schema: {missing}")
