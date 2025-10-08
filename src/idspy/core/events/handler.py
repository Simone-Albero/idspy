from abc import ABC, abstractmethod

from .event import Event


class BaseHandler(ABC):
    """Abstract base class for event handlers."""

    @abstractmethod
    def handle(self, event: Event) -> None:
        """Process the event."""
        pass

    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event. Override for custom logic."""
        return True

    def __call__(self, event: Event) -> None:
        """Make handler callable like the original function-based approach."""
        if self.can_handle(event):
            self.handle(event)
