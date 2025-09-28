from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

from .base import Storage

Key = str
Keys = Sequence[Key]
KVDict = Dict[Key, Any]


@dataclass(frozen=True)
class KeyBinder:
    """Bidirectional key mapper with optional strict mode."""

    mapping: Mapping[Key, Key]
    strict: bool = False

    def __post_init__(self) -> None:
        # Costruisce e valida la mappa inversa (bijettività sul dominio mappato)
        ext2int = dict(self.mapping)
        int2ext: Dict[Key, Key] = {}
        for ext, internal in ext2int.items():
            if internal in int2ext and int2ext[internal] != ext:
                raise ValueError(
                    f"Collisione: la chiave interna {internal!r} è mappata da "
                    f"{int2ext[internal]!r} e {ext!r}"
                )
            int2ext[internal] = ext
        # salva in attributi privati (immutabili a livello dataclass)
        object.__setattr__(self, "_ext2int", ext2int)
        object.__setattr__(self, "_int2ext", int2ext)

    def to_internal(self, key: Key) -> Key:
        """Translate external key to internal."""
        if key in self._ext2int:
            return self._ext2int[key]
        if self.strict:
            raise KeyError(f"Chiave esterna non mappata: {key!r}")
        return key

    def to_external(self, key: Key) -> Key:
        """Translate internal key to external."""
        if key in self._int2ext:
            return self._int2ext[key]
        if self.strict:
            raise KeyError(f"Chiave interna non mappata: {key!r}")
        return key

    def keys_to_internal(self, keys: Iterable[Key]) -> Keys:
        """Translate external keys to internal."""
        return [self.to_internal(k) for k in keys]

    def keys_to_external(self, keys: Iterable[Key]) -> Keys:
        """Translate internal keys to external."""
        return [self.to_external(k) for k in keys]

    def dict_keys_to_internal(self, d: Mapping[Key, Any]) -> KVDict:
        """Translate dict keys from external to internal."""
        return {self.to_internal(k): v for k, v in d.items()}

    def dict_keys_to_external(self, d: Mapping[Key, Any]) -> KVDict:
        """Translate dict keys from internal to external."""
        return {self.to_external(k): v for k, v in d.items()}


class BindedStorage(Storage):
    """Storage that binds external keys to internal keys."""

    def __init__(
        self, storage: Storage, mapping: Mapping[Key, Key], strict: bool = False
    ) -> None:
        self._storage = storage
        self._key_binder = KeyBinder(mapping, strict=strict)

    def get(self, keys: Sequence[str]) -> Dict[str, Any]:
        internal_keys = self._key_binder.keys_to_internal(keys)
        out = self._storage.get(internal_keys)
        return self._key_binder.dict_keys_to_external(out)

    def set(self, values: Mapping[str, Any]) -> None:
        internal_values = self._key_binder.dict_keys_to_internal(values)
        self._storage.set(internal_values)

    def has(self, key: str) -> bool:
        internal_key = self._key_binder.to_internal(key)
        return self._storage.has(internal_key)

    def clear(self) -> None:
        self._storage.clear()

    def delete(self, key: str) -> None:
        internal_key = self._key_binder.to_internal(key)
        self._storage.delete(internal_key)
