from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.location.models import (
    Location,
    RouteComparison,
    TravelModeResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


class GoogleRoutesError(RuntimeError):
    pass


class GoogleRoutesService:
    ENDPOINT = (
        "https://routes.googleapis.com/"
        "directions/v2:computeRoutes"
    )

    MODES = {
        "driving": "DRIVE",
        "walking": "WALK",
        "transit": "TRANSIT",
    }

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        load_dotenv(
            dotenv_path=ENV_PATH,
            override=True,
        )

        self.api_key = (
            api_key
            or os.getenv("GOOGLE_MAPS_API_KEY", "")
        ).strip()

        if not self.api_key:
            raise GoogleRoutesError(
                f"GOOGLE_MAPS_API_KEY is missing from {ENV_PATH}."
            )

        self.timeout_seconds = float(timeout_seconds)
        self.session = requests.Session()

    def compare_routes(
        self,
        start: Location,
        destination: Location,
    ) -> RouteComparison:
        results: dict[str, TravelModeResult] = {}

        for mode in (
            "driving",
            "walking",
            "transit",
        ):
            try:
                results[mode] = self.calculate_route(
                    start=start,
                    destination=destination,
                    mode=mode,
                )
            except GoogleRoutesError as error:
                results[mode] = TravelModeResult(
                    mode=mode,
                    available=False,
                    error=str(error),
                )

        return RouteComparison(
            start=start,
            destination=destination,
            routes=results,
        )

    def calculate_route(
        self,
        start: Location,
        destination: Location,
        mode: str,
    ) -> TravelModeResult:
        normalized = str(mode).strip().lower()
        google_mode = self.MODES.get(normalized)

        if google_mode is None:
            raise GoogleRoutesError(
                f"Unsupported travel mode: {mode}"
            )

        body = {
            "origin": self._waypoint(start),
            "destination": self._waypoint(destination),
            "travelMode": google_mode,
            "languageCode": "en-US",
            "units": "IMPERIAL",
        }

        if google_mode == "TRANSIT":
            body["departureTime"] = (
                datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "routes.duration,"
                "routes.distanceMeters,"
                "routes.polyline.encodedPolyline,"
                "routes.description"
            ),
        }

        try:
            response = self.session.post(
                self.ENDPOINT,
                json=body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise GoogleRoutesError(
                f"Google Routes request failed: {error}"
            ) from error

        if not response.ok:
            try:
                message = (
                    response.json()
                    .get("error", {})
                    .get("message", response.text)
                )
            except ValueError:
                message = response.text

            raise GoogleRoutesError(
                f"{normalized.title()} route failed: {message}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise GoogleRoutesError(
                "Google Routes returned invalid JSON."
            ) from error

        routes = payload.get("routes", [])

        if not routes:
            raise GoogleRoutesError(
                f"No {normalized} route is available."
            )

        route = routes[0]

        return TravelModeResult(
            mode=normalized,
            available=True,
            distance_meters=float(
                route.get("distanceMeters", 0.0)
            ),
            duration_seconds=self._duration_seconds(
                route.get("duration", "0s")
            ),
            geometry=self._decode_polyline(
                route.get("polyline", {})
                .get("encodedPolyline", "")
            ),
            description=str(
                route.get("description", "")
            ),
        )

    @staticmethod
    def _waypoint(location: Location) -> dict:
        return {
            "location": {
                "latLng": {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                }
            }
        }

    @staticmethod
    def _duration_seconds(value: str) -> float:
        text = str(value).strip()
        if text.endswith("s"):
            text = text[:-1]
        try:
            return float(text)
        except ValueError:
            return 0.0

    @staticmethod
    def _decode_polyline(
        encoded: str,
    ) -> list[list[float]]:
        if not encoded:
            return []

        points: list[list[float]] = []
        index = 0
        latitude = 0
        longitude = 0

        while index < len(encoded):
            lat_change, index = (
                GoogleRoutesService._decode_value(
                    encoded,
                    index,
                )
            )
            lon_change, index = (
                GoogleRoutesService._decode_value(
                    encoded,
                    index,
                )
            )

            latitude += lat_change
            longitude += lon_change

            points.append(
                [
                    latitude / 100000.0,
                    longitude / 100000.0,
                ]
            )

        return points

    @staticmethod
    def _decode_value(
        encoded: str,
        index: int,
    ) -> tuple[int, int]:
        result = 0
        shift = 0

        while True:
            value = ord(encoded[index]) - 63
            index += 1
            result |= (value & 0x1F) << shift
            shift += 5
            if value < 0x20:
                break

        decoded = (
            ~(result >> 1)
            if result & 1
            else result >> 1
        )

        return decoded, index