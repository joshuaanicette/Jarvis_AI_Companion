from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class VisionFrame:
    image: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    objects: list[dict] = field(default_factory=list)
    faces: list[dict] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    qr_codes: list[str] = field(default_factory=list)
    scene: str = ""
    metadata: dict = field(default_factory=dict)
