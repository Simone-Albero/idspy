from typing import Any, Dict, Mapping, Sequence

from .base import Storage, Port


class DictStorage(Storage):
    """A Storage implementation that uses a plain dictionary internally."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def get(self, ports: Sequence[Port]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for port in ports:
            if port.name not in self._data:
                raise KeyError(f"Port '{port.name}' not found in storage.")
            value = self._data[port.name]
            if not isinstance(value, port.dtype) and port.dtype is not Any:
                raise TypeError(
                    f"Type mismatch for port '{port.name}': expected {port.dtype}, got {type(value)}"
                )
            result[port.name] = value
        return result

    def set(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if name in self._data:
                existing_value = self._data[name]
                if type(existing_value) != type(value):
                    raise TypeError(
                        f"Type mismatch for port '{name}': existing type {type(existing_value)}, new type {type(value)}"
                    )

            self._data[name] = value

    def has(self, name: str) -> bool:
        return name in self._data

    def clear(self) -> None:
        self._data.clear()

    def delete(self, name: str) -> None:
        if name in self._data:
            del self._data[name]
