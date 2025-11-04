from abc import abstractmethod, ABC
from typing import Dict, Any, Optional

from .base import Step


class ConditionalStep(Step, ABC):
    """A Step that can be conditionally executed."""

    def __init__(
        self, *args, step: Optional[Step] = None, name: Optional[str] = None, **kwargs
    ):
        if step is not None:
            self._controlled_step = step
            super().__init__(name=name or f"conditional_{step.name}", **kwargs)
        else:
            self._controlled_step = None
            super().__init__(*args, **kwargs)

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

        if self._controlled_step is not None:
            return self._controlled_step.compute(**kwargs)

        return self.compute(**kwargs)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(")")
        return f"{base}, conditional=True)"
