from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Dict

from ...common.predicate import Predicate

StoragePredicate = Predicate["Storage"]


class Storage(ABC):
    """Abstract persistent storage for reading/writing values by key."""

    @abstractmethod
    def get(self, keys: Sequence[str]) -> Dict[str, Any]:
        """Return a mapping {key: value} for the requested keys."""
        raise NotImplementedError

    @abstractmethod
    def set(self, values: Mapping[str, Any]) -> None:
        """Persist a mapping {key: value}. Type checks may occur at this layer."""
        raise NotImplementedError

    @abstractmethod
    def has(self, key: str) -> bool:
        """Return True if a value with the given key exists in storage."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored values."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a stored value by key."""
        raise NotImplementedError


def has_key(key: str) -> StoragePredicate:
    """Accept states that contain `key`."""
    return lambda storage: storage.has(key)


def has_keys(keys: Sequence[str]) -> StoragePredicate:
    """Accept states that contain all `keys`."""
    return lambda storage: all(storage.has(k) for k in keys)


def lacks_key(key: str) -> StoragePredicate:
    """Accept states that do not contain `key`."""
    return lambda storage: not storage.has(key)


def lacks_keys(keys: Sequence[str]) -> StoragePredicate:
    """Accept states that do not contain any of `keys`."""
    return lambda storage: all(not storage.has(k) for k in keys)
