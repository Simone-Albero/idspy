from abc import abstractmethod, ABC
from typing import Dict, Any

from .base import Step


class ConditionalStep(Step, ABC):
    """A Step that can be conditionally executed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._should_run = True

    @abstractmethod
    def should_run(self, **kwargs) -> bool:
        """Return True to run, False to skip."""
        ...

    @abstractmethod
    def on_skip(self, **kwargs) -> None:
        """Called if the step is skipped."""
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the step if the condition holds, otherwise skip.
        kwargs are unpacked by port name.
        Defaults declared in the step signature will be used if storage
        does not provide the value.
        """
        if not self.should_run(**kwargs):
            self.on_skip(**kwargs)
            return {}
        return self.compute(**kwargs)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, should_run={self._should_run})"
