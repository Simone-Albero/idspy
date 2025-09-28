from typing import Any, Dict, Mapping, Optional, Sequence

from .base import Storage


class DictStorage(Storage):
    """A Storage implementation that uses a plain dictionary internally."""

    def __init__(self, data: Optional[Mapping[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = dict(data) if data else {}

    def get(self, keys: Sequence[str]) -> Dict[str, Any]:
        return {k: self._data[k] for k in keys if k in self._data}

    def set(self, values: Mapping[str, Any]) -> None:
        self._data.update(values)

    def has(self, name: str) -> bool:
        return name in self._data

    def clear(self) -> None:
        self._data.clear()

    def delete(self, name: str) -> None:
        if name in self._data:
            del self._data[name]
