from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.location.google_geocoder import GoogleGeocoder
from src.location.google_routes_service import (
    GoogleRoutesService,
)
from src.location.models import Location
from src.location.route_service import OSRMRouteService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


class LocationManager:
    def __init__(
        self,
        google_geocoder: GoogleGeocoder | None = None,
        google_routes: GoogleRoutesService | None = None,
        route_service: OSRMRouteService | None = None,
    ):
        load_dotenv(
            dotenv_path=ENV_PATH,
            override=True,
        )

        self.google_geocoder = (
            google_geocoder or GoogleGeocoder()
        )
        self.google_routes = (
            google_routes or GoogleRoutesService()
        )
        self.route_service = (
            route_service or OSRMRouteService()
        )

        self.latest_route = None
        self.latest_comparison = None
        self._home_location: Location | None = None

    def get_home_location(
        self,
        refresh: bool = False,
    ) -> Location:
        if (
            self._home_location is not None
            and not refresh
        ):
            return Location(
                self._home_location.latitude,
                self._home_location.longitude,
                self._home_location.label,
            )

        home_address = os.getenv(
            "HOME_ADDRESS",
            "",
        ).strip()

        if not home_address:
            raise ValueError(
                f"HOME_ADDRESS is missing from {ENV_PATH}."
            )

        home = self.google_geocoder.geocode_address(
            home_address,
            require_exact=True,
        )

        self._home_location = home

        return Location(
            home.latitude,
            home.longitude,
            home.label,
        )

    def route_between_addresses(
        self,
        start_query: str,
        destination_query: str,
    ):
        start = self.google_geocoder.geocode_address(
            start_query
        )
        destination = (
            self.google_geocoder.geocode_address(
                destination_query
            )
        )

        route = self.route_service.calculate_route(
            start=start,
            destination=destination,
        )

        self.latest_route = route
        return route

    def route_from_coordinates(
        self,
        latitude: float,
        longitude: float,
        destination_query: str,
        start_label: str = "Current location",
    ):
        self._validate_coordinates(
            latitude,
            longitude,
        )

        start = Location(
            float(latitude),
            float(longitude),
            start_label,
        )

        destination = (
            self.google_geocoder.geocode_address(
                destination_query
            )
        )

        route = self.route_service.calculate_route(
            start=start,
            destination=destination,
        )

        self.latest_route = route
        return route

    def compare_home_to_destination(
        self,
        destination_query: str,
    ):
        point_a = self.get_home_location()
        point_b = self.google_geocoder.geocode_address(
            destination_query
        )

        comparison = self.google_routes.compare_routes(
            start=point_a,
            destination=point_b,
        )

        self.latest_comparison = comparison
        return comparison

    def compare_coordinates_to_destination(
        self,
        latitude: float,
        longitude: float,
        destination_query: str,
        start_label: str = "Current location",
    ):
        self._validate_coordinates(
            latitude,
            longitude,
        )

        point_a = Location(
            float(latitude),
            float(longitude),
            start_label,
        )

        point_b = self.google_geocoder.geocode_address(
            destination_query
        )

        comparison = self.google_routes.compare_routes(
            start=point_a,
            destination=point_b,
        )

        self.latest_comparison = comparison
        return comparison

    @staticmethod
    def _validate_coordinates(
        latitude: float,
        longitude: float,
    ) -> None:
        latitude = float(latitude)
        longitude = float(longitude)

        if not -90 <= latitude <= 90:
            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not -180 <= longitude <= 180:
            raise ValueError(
                "Longitude must be between -180 and 180."
            )