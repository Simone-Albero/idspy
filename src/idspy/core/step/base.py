from abc import ABC, abstractmethod
from typing import Any, Sequence, Dict, Optional


class Step(ABC):
    """Abstract unit of work with typed inputs and outputs."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @classmethod
    def needs(cls, *requirements: str):
        def decorator(subcls):
            subcls._requires = tuple(requirements)
            return subcls

        return decorator

    @property
    def requires(self) -> Sequence[str]:
        return getattr(self.__class__, "_requires", ())

    @abstractmethod
    def bindings(self) -> Dict[str, str]:
        """Return a local-to-storage key mapping."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, **kwargs: Any) -> Dict[str, Any]:
        """Compute the step's outputs from its inputs.
        Rules:
        - kwargs are unpacked by local names.
        - Defaults declared in the step signature will be used if storage
          does not provide the value.
        - Return a dict mapping output local names to values.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Dict[str, Any]:
        return self.run(*args, **kwds)

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the step."""
        return self.compute(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
