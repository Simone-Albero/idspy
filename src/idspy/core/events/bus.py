import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Final, Union, Any

from .event import Event, EventPredicate
from .handler import BaseHandler

Handler = Union[BaseHandler, Callable[[Event], None]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Entry:
    """Subscription entry."""

    callback: Handler
    predicate: Optional[EventPredicate]
    token: int  # unique id for unsubscription
    priority: int


class EventBus:
    """"""

    ALL: Final[Optional[str]] = None  # subscribe to all events

    def __init__(self) -> None:
        self._subs: Dict[Optional[str], List[_Entry]] = {}
        self._next_token: int = 1

    def subscribe(
        self,
        callback: Handler,
        event_type: Optional[str] = None,
        predicate: Optional[EventPredicate] = None,
        priority: int = 1,
    ) -> int:
        """Register callback; returns a token."""
        token = self._next_token
        self._next_token += 1
        self._subs.setdefault(event_type, []).append(
            _Entry(callback, predicate, token, priority)
        )
        return token

    def on(
        self,
        event_type: Optional[str] = None,
        predicate: Optional[EventPredicate] = None,
        priority: int = 1,
    ) -> Callable[[Handler], Handler]:
        """Decorator to subscribe a handler."""

        def decorator(fn: Handler) -> Handler:
            self.subscribe(
                fn, event_type=event_type, predicate=predicate, priority=priority
            )
            return fn

        return decorator

    def unsubscribe(self, token: int) -> bool:
        """Unsubscribe by token; returns True if removed."""
        removed = False
        for key, entries in list(self._subs.items()):
            kept = [e for e in entries if e.token != token]
            if len(kept) != len(entries):
                removed = True
                if kept:
                    self._subs[key] = kept
                else:
                    self._subs.pop(key, None)
        return removed

    def publish(
        self, source: str, payload: Dict[str, Any], event_type: Optional[str] = None
    ) -> None:
        """Dispatch event to subscribers; log handler errors."""
        targets: List[_Entry] = []
        targets.extend(self._subs.get(event_type, ()))
        targets.extend(self._subs.get(self.ALL, ()))

        targets.sort(key=lambda entry: entry.priority)
        event = Event(type=event_type or "unknown", source=source, payload=payload)

        for entry in list(targets):
            try:
                if entry.predicate is None or entry.predicate(event):
                    entry.callback(event)
            except Exception:
                logger.exception(
                    "Subscriber error for %s (id=%s)", event.type, event.id
                )
