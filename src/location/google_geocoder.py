from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.location.models import Location


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


class GoogleGeocodingError(RuntimeError):
    pass


class GoogleGeocoder:
    ENDPOINT = (
        "https://maps.googleapis.com/"
        "maps/api/geocode/json"
    )

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
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
            raise GoogleGeocodingError(
                f"GOOGLE_MAPS_API_KEY is missing from {ENV_PATH}."
            )

        self.timeout_seconds = float(timeout_seconds)
        self.session = requests.Session()
        self._cache: dict[str, Location] = {}

    def geocode_address(
        self,
        address: str,
        require_exact: bool = False,
    ) -> Location:
        cleaned = str(address).strip()

        if not cleaned:
            raise GoogleGeocodingError(
                "An address or place name is required."
            )

        key = f"{cleaned.casefold()}:{require_exact}"

        if key in self._cache:
            cached = self._cache[key]
            return Location(
                cached.latitude,
                cached.longitude,
                cached.label,
            )

        try:
            response = self.session.get(
                self.ENDPOINT,
                params={
                    "address": cleaned,
                    "region": "us",
                    "key": self.api_key,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise GoogleGeocodingError(
                f"Google Geocoding request failed: {error}"
            ) from error
        except ValueError as error:
            raise GoogleGeocodingError(
                "Google Geocoding returned invalid JSON."
            ) from error

        status = str(payload.get("status", ""))

        if status != "OK":
            message = payload.get(
                "error_message",
                status or "Unknown error",
            )
            raise GoogleGeocodingError(
                f"Google Geocoding failed: {message}"
            )

        results = payload.get("results", [])

        if not results:
            raise GoogleGeocodingError(
                f"No address matched '{cleaned}'."
            )

        result = results[0]

        if require_exact and result.get("partial_match", False):
            raise GoogleGeocodingError(
                "Google returned only a partial address match."
            )

        coordinates = (
            result.get("geometry", {})
            .get("location", {})
        )

        try:
            latitude = float(coordinates["lat"])
            longitude = float(coordinates["lng"])
        except (KeyError, TypeError, ValueError) as error:
            raise GoogleGeocodingError(
                "Google returned invalid coordinates."
            ) from error

        location = Location(
            latitude=latitude,
            longitude=longitude,
            label=str(
                result.get(
                    "formatted_address",
                    cleaned,
                )
            ),
        )

        self._cache[key] = location

        return Location(
            location.latitude,
            location.longitude,
            location.label,
        )