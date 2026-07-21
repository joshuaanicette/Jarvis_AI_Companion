from __future__ import annotations

import logging
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.location.location_manager import LocationManager


logger = logging.getLogger("jay.navigation")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = (
    PROJECT_ROOT
    / "templates"
    / "navigation_dashboard.html"
)
STATIC_PATH = (
    PROJECT_ROOT
    / "static"
    / "navigation"
)


class ManualRouteRequest(BaseModel):
    start: str = Field(min_length=2, max_length=300)
    destination: str = Field(min_length=2, max_length=300)


class HomeComparisonRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=300)


class BrowserComparisonRequest(BaseModel):
    latitude: float
    longitude: float
    destination: str = Field(min_length=2, max_length=300)


class NavigationDashboard:
    def __init__(
        self,
        location_manager: LocationManager,
        host: str = "127.0.0.1",
        port: int = 8770,
    ):
        self.location_manager = location_manager
        self.host = host
        self.port = int(port)
        self.running = False
        self._thread = None
        self._server = None

        self.app = FastAPI(
            title="Jay Navigation Dashboard"
        )

        STATIC_PATH.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.app.mount(
            "/static/navigation",
            StaticFiles(
                directory=str(STATIC_PATH)
            ),
            name="navigation_static",
        )

        self._configure_routes()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def _configure_routes(self) -> None:
        @self.app.get(
            "/",
            response_class=HTMLResponse,
        )
        def dashboard() -> HTMLResponse:
            if not TEMPLATE_PATH.exists():
                return HTMLResponse(
                    "<h1>Navigation dashboard template missing.</h1>",
                    status_code=500,
                )

            return HTMLResponse(
                TEMPLATE_PATH.read_text(
                    encoding="utf-8"
                )
            )

        @self.app.get("/api/status")
        def status() -> dict:
            return {
                "online": True,
                "supports": [
                    "manual-driving",
                    "home-comparison",
                    "current-comparison",
                ],
            }

        @self.app.post("/api/route/manual")
        def manual_route(
            request: ManualRouteRequest,
        ) -> dict:
            try:
                route = (
                    self.location_manager
                    .route_between_addresses(
                        request.start,
                        request.destination,
                    )
                )
                return route.to_dict()
            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error),
                ) from error

        @self.app.post(
            "/api/route/compare/home"
        )
        def compare_home(
            request: HomeComparisonRequest,
        ) -> dict:
            try:
                comparison = (
                    self.location_manager
                    .compare_home_to_destination(
                        request.destination
                    )
                )
                return comparison.to_dict()
            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error),
                ) from error

        @self.app.post(
            "/api/route/compare/browser"
        )
        def compare_browser(
            request: BrowserComparisonRequest,
        ) -> dict:
            try:
                comparison = (
                    self.location_manager
                    .compare_coordinates_to_destination(
                        latitude=request.latitude,
                        longitude=request.longitude,
                        destination_query=request.destination,
                    )
                )
                return comparison.to_dict()
            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error),
                ) from error

    def start(self) -> bool:
        if self.running:
            return False

        self.running = True

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )

        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
            name="JayNavigationDashboard",
        )

        self._thread.start()
        time.sleep(1.0)

        logger.info(
            "Navigation dashboard started at %s",
            self.url,
        )

        return True

    def open(self) -> str:
        if not self.running:
            self.start()

        webbrowser.open(self.url)
        return "I opened the navigation dashboard."

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

        self.running = False