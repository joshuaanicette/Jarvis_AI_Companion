from __future__ import annotations

import requests

from src.location.models import Location, RouteResult


class RoutingError(RuntimeError):
    pass


class OSRMRouteService:
    def __init__(
        self,
        endpoint: str = "https://router.project-osrm.org",
        timeout_seconds: float = 15.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.session = requests.Session()

    def calculate_route(
        self,
        start: Location,
        destination: Location,
        profile: str = "driving",
    ) -> RouteResult:
        coordinates = (
            f"{start.longitude},{start.latitude};"
            f"{destination.longitude},{destination.latitude}"
        )

        url = (
            f"{self.endpoint}/route/v1/"
            f"{profile}/{coordinates}"
        )

        try:
            response = self.session.get(
                url,
                params={
                    "overview": "full",
                    "geometries": "geojson",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise RoutingError(
                f"OSRM request failed: {error}"
            ) from error

        routes = payload.get("routes", [])

        if payload.get("code") != "Ok" or not routes:
            raise RoutingError(
                "OSRM did not return a route."
            )

        route = routes[0]
        geometry = [
            [float(point[1]), float(point[0])]
            for point in (
                route.get("geometry", {})
                .get("coordinates", [])
            )
            if len(point) >= 2
        ]

        return RouteResult(
            start=start,
            destination=destination,
            distance_meters=float(
                route.get("distance", 0.0)
            ),
            duration_seconds=float(
                route.get("duration", 0.0)
            ),
            geometry=geometry,
            profile=profile,
        )