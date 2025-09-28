from abc import ABC, abstractmethod
from typing import Any, Sequence, Dict, Optional


class Step(ABC):
    """Abstract unit of work with typed inputs and outputs."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @classmethod
    def requires(cls, *requirements: str):
        def decorator(subcls):
            subcls._requires = tuple(requirements)
            return subcls

        return decorator

    @property
    def requires(self) -> Sequence[str]:
        return getattr(self.__class__, "_requires", ())

    @property
    @abstractmethod
    def get_bindings(self) -> Dict[str, str]:
        """Return a mapping of port name to bound key, if any."""
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the step.
        kwargs are unpacked by port name.
        Defaults declared in the step signature will be used if storage
        does not provide the value.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Dict[str, Any]:
        return self.run(*args, **kwds)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
