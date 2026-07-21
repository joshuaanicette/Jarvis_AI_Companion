from dataclasses import dataclass, field
from typing import Any


@dataclass
class JayState:
    online: bool = False
    mode: str = "idle"
    battery: float = 100.0
    listening: bool = False
    thinking: bool = False
    speaking: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def update(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.data[key] = value

    def get(self, key: str, default=None):
        return getattr(self, key, self.data.get(key, default))
