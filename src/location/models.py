from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Location:
    latitude: float
    longitude: float
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteResult:
    start: Location
    destination: Location
    distance_meters: float
    duration_seconds: float
    geometry: list[list[float]]
    profile: str = "driving"

    @property
    def distance_miles(self) -> float:
        return self.distance_meters / 1609.344

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "destination": self.destination.to_dict(),
            "distance_meters": self.distance_meters,
            "distance_miles": round(self.distance_miles, 2),
            "duration_seconds": self.duration_seconds,
            "duration_minutes": round(self.duration_minutes, 1),
            "geometry": self.geometry,
            "profile": self.profile,
        }


@dataclass(slots=True)
class TravelModeResult:
    mode: str
    available: bool
    distance_meters: float = 0.0
    duration_seconds: float = 0.0
    geometry: list[list[float]] = field(default_factory=list)
    description: str = ""
    error: str = ""

    @property
    def distance_miles(self) -> float:
        return self.distance_meters / 1609.344

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "available": self.available,
            "distance_meters": self.distance_meters,
            "distance_miles": round(self.distance_miles, 2),
            "duration_seconds": self.duration_seconds,
            "duration_minutes": round(self.duration_minutes, 1),
            "geometry": self.geometry,
            "description": self.description,
            "error": self.error,
        }


@dataclass(slots=True)
class RouteComparison:
    start: Location
    destination: Location
    routes: dict[str, TravelModeResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "destination": self.destination.to_dict(),
            "routes": {
                mode: route.to_dict()
                for mode, route in self.routes.items()
            },
        }