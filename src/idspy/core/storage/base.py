from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Dict, Type

from ...common.predicate import Predicate

StoragePredicate = Predicate["Storage"]


@dataclass(frozen=True)
class Port:
    """A logical, typed channel for data."""

    name: str
    dtype: Type[Any]  # use `typing.Any` to opt out of checks

    def __hash__(self) -> int:
        # Ports are uniquely identified by name; conflicting types with same name are disallowed elsewhere.
        return hash(self.name)


class Storage(ABC):
    """Abstract persistent storage for reading/writing values by Port."""

    @abstractmethod
    def get(self, ports: Sequence[Port]) -> Dict[str, Any]:
        """Return a mapping {port.name: value} for the requested ports."""
        raise NotImplementedError

    @abstractmethod
    def set(self, values: Mapping[str, Any]) -> None:
        """Persist a mapping {name: value}. Type checks may occur at this layer."""
        raise NotImplementedError

    @abstractmethod
    def has(self, name: str) -> bool:
        """Return True if a value with the given name exists in storage."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all stored values."""
        raise NotImplementedError

    def delete(self, name: str) -> None:
        """Delete a stored value by name."""
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


def key_equals(key: str, value: Any) -> StoragePredicate:
    """Accept states where `key` exists and equals `value`."""
    return (
        lambda storage: storage.has(key) and storage.get([Port(key, Any)])[key] == value
    )


def key_not_equals(key: str, value: Any) -> StoragePredicate:
    """Accept states where `key` does not exist or does not equal `value`."""
    return (
        lambda storage: not storage.has(key)
        or storage.get([Port(key, Any)])[key] != value
    )


def key_is_truthy(key: str) -> StoragePredicate:
    """Accept states where `key` exists and is truthy."""
    return lambda storage: storage.has(key) and bool(storage.get([Port(key, Any)])[key])


def key_is_falsy(key: str) -> StoragePredicate:
    """Accept states where `key` is missing or falsy."""
    return lambda storage: not storage.has(key) or not bool(
        storage.get([Port(key, Any)])[key]
    )
