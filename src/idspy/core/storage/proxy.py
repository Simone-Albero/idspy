from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

from .base import Storage

Key = str
Keys = Sequence[Key]
KVDict = Dict[Key, Any]


@dataclass(frozen=True)
class KeyBinder:
    """Maps external keys to internal keys with optional strict validation."""

    mapping: Mapping[Key, Key]
    strict: bool = False

    def __post_init__(self) -> None:
        """Build and validate inverse mapping for bijection."""
        ext2int = dict(self.mapping)
        int2ext: Dict[Key, Key] = {}

        for ext, internal in ext2int.items():
            if internal in int2ext and int2ext[internal] != ext:
                raise ValueError(
                    f"Key collision: internal key {internal!r} is mapped from "
                    f"{int2ext[internal]!r} and {ext!r}"
                )
            int2ext[internal] = ext

        object.__setattr__(self, "_ext2int", ext2int)
        object.__setattr__(self, "_int2ext", int2ext)

    def to_internal(self, key: Key) -> Key:
        """Convert external key to internal key."""
        if key in self._ext2int:
            return self._ext2int[key]
        if self.strict:
            raise KeyError(f"Unmapped external key: {key!r}")
        return key

    def to_external(self, key: Key) -> Key:
        """Convert internal key to external key."""
        if key in self._int2ext:
            return self._int2ext[key]
        if self.strict:
            raise KeyError(f"Unmapped internal key: {key!r}")
        return key

    def keys_to_internal(self, keys: Iterable[Key]) -> Keys:
        """Convert external keys to internal keys."""
        return [self.to_internal(k) for k in keys]

    def keys_to_external(self, keys: Iterable[Key]) -> Keys:
        """Convert internal keys to external keys."""
        return [self.to_external(k) for k in keys]

    def dict_keys_to_internal(self, d: Mapping[Key, Any]) -> KVDict:
        """Convert dict keys from external to internal."""
        return {self.to_internal(k): v for k, v in d.items()}

    def dict_keys_to_external(self, d: Mapping[Key, Any]) -> KVDict:
        """Convert dict keys from internal to external."""
        return {self.to_external(k): v for k, v in d.items()}


class BindedStorage(Storage):
    """Storage wrapper that translates between external and internal keys."""

    def __init__(
        self, storage: Storage, mapping: Mapping[Key, Key], strict: bool = False
    ) -> None:
        """Initialize with underlying storage and key mapping."""
        self._storage = storage
        self._key_binder = KeyBinder(mapping, strict=strict)

    def get(self, keys: Sequence[str]) -> Dict[str, Any]:
        """Get values by external keys."""
        internal_keys = self._key_binder.keys_to_internal(keys)
        result = self._storage.get(internal_keys)
        return self._key_binder.dict_keys_to_external(result)

    def set(self, values: Mapping[str, Any]) -> None:
        """Set values using external keys."""
        internal_values = self._key_binder.dict_keys_to_internal(values)
        self._storage.set(internal_values)

    def has(self, key: str) -> bool:
        """Check if external key exists."""
        internal_key = self._key_binder.to_internal(key)
        return self._storage.has(internal_key)

    def clear(self) -> None:
        """Clear all stored data."""
        self._storage.clear()

    def delete(self, key: str) -> None:
        """Delete value by external key."""
        internal_key = self._key_binder.to_internal(key)
        self._storage.delete(internal_key)

    def as_dict(self) -> Dict[str, Any]:
        """Return all data as dict with external keys."""
        internal_dict = self._storage.as_dict()
        return self._key_binder.dict_keys_to_external(internal_dict)
