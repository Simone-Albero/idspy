from abc import abstractmethod, ABC
from typing import Any, Dict

from .base import Step


class FittableStep(Step, ABC):
    """A Step that can be fitted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the step has executed its fit hook at least once."""
        return self._is_fitted

    @abstractmethod
    def fit_impl(self, **kwargs) -> None:
        """Actual fitting logic to be implemented by subclasses."""
        raise NotImplementedError

    def fit(self, **kwargs) -> None:
        """
        Fit the step.
        kwargs are unpacked by port name.
        Defaults declared in the step signature will be used if storage
        does not provide the value.
        """
        self._is_fitted = True
        self.fit_impl(**kwargs)

    def run(self, **kwargs) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError(f"Step {self} is not fitted; cannot run.")
        return super().run(**kwargs)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, fitted={self._is_fitted})"
