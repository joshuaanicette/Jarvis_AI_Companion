from __future__ import annotations

import time
from threading import Lock

import requests

from src.location.models import Location


class GeocodingError(RuntimeError):
    pass


class NominatimGeocoder:
    def __init__(
        self,
        user_agent: str = "JayAICompanion/1.0",
        endpoint: str = "https://nominatim.openstreetmap.org/search",
        timeout_seconds: float = 12.0,
        minimum_interval_seconds: float = 1.0,
    ):
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)
        self.minimum_interval_seconds = max(
            1.0,
            float(minimum_interval_seconds),
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )

        self._request_lock = Lock()
        self._last_request_at = 0.0
        self._cache: dict[str, Location] = {}

    def geocode(self, query: str) -> Location:
        cleaned_query = str(query).strip()

        if not cleaned_query:
            raise GeocodingError(
                "A starting location or destination is required."
            )

        cache_key = cleaned_query.casefold()

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return Location(
                latitude=cached.latitude,
                longitude=cached.longitude,
                label=cached.label,
            )

        with self._request_lock:
            elapsed = time.monotonic() - self._last_request_at

            if elapsed < self.minimum_interval_seconds:
                time.sleep(
                    self.minimum_interval_seconds - elapsed
                )

            try:
                response = self.session.get(
                    self.endpoint,
                    params={
                        "q": cleaned_query,
                        "format": "jsonv2",
                        "limit": 1,
                        "addressdetails": 1,
                    },
                    timeout=self.timeout_seconds,
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                results = response.json()

            except requests.RequestException as error:
                raise GeocodingError(
                    f"The geocoding service could not be reached: {error}"
                ) from error

            except ValueError as error:
                raise GeocodingError(
                    "The geocoding service returned invalid data."
                ) from error

        if not isinstance(results, list) or not results:
            raise GeocodingError(
                f"I could not find a location matching '{cleaned_query}'."
            )

        result = results[0]

        try:
            location = Location(
                latitude=float(result["lat"]),
                longitude=float(result["lon"]),
                label=str(
                    result.get(
                        "display_name",
                        cleaned_query,
                    )
                ),
            )

        except (KeyError, TypeError, ValueError) as error:
            raise GeocodingError(
                "The geocoding result did not contain valid coordinates."
            ) from error

        self._cache[cache_key] = location

        return Location(
            latitude=location.latitude,
            longitude=location.longitude,
            label=location.label,
        )
