import threading
import time
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

from src.core.logger import logger
from src.tools.weather_tool import (
    WeatherTool,
    WeatherToolError,
)
from src.weather.clothing_advisor import (
    ClothingAdvisor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = (
    PROJECT_ROOT
    / "templates"
    / "weather_dashboard.html"
)


class WeatherDashboard:
    def __init__(
        self,
        weather_tool: WeatherTool,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.weather_tool = weather_tool
        self.clothing_advisor = ClothingAdvisor()

        self.host = host
        self.port = port

        self.app = FastAPI(
            title="Jay Weather Dashboard"
        )

        self.running = False
        self._thread = None
        self._configure_routes()

    def _configure_routes(self) -> None:
        @self.app.get(
            "/",
            response_class=HTMLResponse,
        )
        def dashboard():
            if not TEMPLATE_PATH.exists():
                return HTMLResponse(
                    "<h1>Weather dashboard template missing.</h1>",
                    status_code=500,
                )

            return TEMPLATE_PATH.read_text(
                encoding="utf-8"
            )

        @self.app.get("/api/weather/{city}")
        def weather_api(city: str):
            try:
                forecast = (
                    self.weather_tool.get_forecast(
                        city=city,
                    )
                )

                current = (
                    self.weather_tool.get_current_data(
                        city=city,
                    )
                )

                clothing = (
                    self.clothing_advisor.recommend(
                        current
                    )
                )

                return {
                    "current": current,
                    "forecast": forecast,
                    "clothing": clothing,
                }

            except WeatherToolError as error:
                raise HTTPException(
                    status_code=502,
                    detail=str(error),
                ) from error

    def start(self) -> bool:
        if self.running:
            return False

        self.running = True

        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="JayWeatherDashboard",
        )

        self._thread.start()
        time.sleep(1.0)

        logger.info(
            "Weather dashboard started at http://%s:%d",
            self.host,
            self.port,
        )

        return True

    def _run_server(self) -> None:
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )

    def open_city(self, city: str) -> str:
        if not self.running:
            self.start()

        dashboard_url = (
            f"http://{self.host}:{self.port}"
            f"/?city={city}"
        )

        webbrowser.open(dashboard_url)

        return (
            f"I opened the weather dashboard for {city}."
        )