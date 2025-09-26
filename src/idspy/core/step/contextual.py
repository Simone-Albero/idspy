from abc import abstractmethod
from typing import Dict, Any

from .base import Step


class ContextualStep(Step):
    """A Step that runs within a context manager."""

    @abstractmethod
    def context(self, **kwargs) -> Any:
        """Return a context manager."""
        ...

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the step within the provided context.
        kwargs are unpacked by port name.
        Defaults declared in the step signature will be used if storage
        does not provide the value.
        """
        with self.context(**kwargs) as ctx:
            return super().run(context=ctx, **kwargs)

    def __repr__(self):
        base = super().__repr__().rstrip(")")
        return f"{base}, context={self.context})"
