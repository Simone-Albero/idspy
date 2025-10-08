from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Dict

from ...common.predicate import Predicate

EventPredicate = Predicate["Event"]


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event."""

    type: str
    source: str
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "payload": dict(self.payload),
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        keys = list(self.payload)
        head = keys[:5]
        more = f", +{len(keys) - 5} keys" if len(keys) > 5 else ""
        return (
            f"Event({self.type!r}, source={self.source!r}, payload_keys={head!r}{more})"
        )


def only_source(event_source: str) -> EventPredicate:
    """Accept only events with a specific source (pipeline/step)."""
    return lambda e: e.source == event_source


def source_startswith(prefix: str) -> EventPredicate:
    """Accept events whose source starts with the given prefix."""
    return lambda e: e.source.startswith(prefix)


def has_payload_key(key: str) -> EventPredicate:
    """Accept events that carry a certain payload key."""
    return lambda e: key in e.payload


def payload_key_equals(key: str, value: Any) -> EventPredicate:
    """Accept events whose payload contains a key with a specific value."""
    return lambda e: e.payload.get(key) == value
