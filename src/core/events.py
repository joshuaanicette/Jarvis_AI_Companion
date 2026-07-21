from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class Event:
    name: str
    data: Any = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
