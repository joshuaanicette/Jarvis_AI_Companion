import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.tools.tool import Tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


class WeatherToolError(RuntimeError):
    """Raised when weather data cannot be retrieved."""


class WeatherTool(Tool):
    CURRENT_URL = (
        "https://api.openweathermap.org/data/2.5/weather"
    )

    FORECAST_URL = (
        "https://api.openweathermap.org/data/2.5/forecast"
    )

    def __init__(
        self,
        api_key: str | None = None,
        default_city: str | None = None,
        default_country: str | None = None,
        timeout: float = 10.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("OPENWEATHER_API_KEY")
        )

        self.default_city = (
            default_city
            or os.getenv("JAY_WEATHER_CITY")
            or "Philadelphia"
        )

        self.default_country = (
            default_country
            or os.getenv("JAY_WEATHER_COUNTRY")
            or "US"
        )

        self.timeout = timeout

    @property
    def name(self) -> str:
        return "weather"

    def execute(
        self,
        city: str | None = None,
        country: str | None = None,
        include_forecast: bool = False,
    ):
        selected_city = city or self.default_city
        selected_country = (
            country or self.default_country
        )

        if include_forecast:
            return self.get_forecast(
                selected_city,
                selected_country,
            )

        return self.get_current_summary(
            selected_city,
            selected_country,
        )

    def get_current_data(
        self,
        city: str,
        country: str = "US",
    ) -> dict[str, Any]:
        self._validate_key()

        response = self._request(
            self.CURRENT_URL,
            {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "units": "imperial",
            },
        )

        return response

    def get_forecast(
        self,
        city: str,
        country: str = "US",
    ) -> dict[str, Any]:
        self._validate_key()

        data = self._request(
            self.FORECAST_URL,
            {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "units": "imperial",
            },
        )

        return self._normalize_forecast(data)

    def get_current_summary(
        self,
        city: str,
        country: str = "US",
    ) -> str:
        data = self.get_current_data(city, country)

        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        weather_items = data.get("weather") or [{}]

        temperature = float(main.get("temp", 0))
        feels_like = float(
            main.get("feels_like", temperature)
        )
        humidity = int(main.get("humidity", 0))
        wind_speed = float(wind.get("speed", 0))
        cloud_cover = int(clouds.get("all", 0))

        condition = str(
            weather_items[0].get(
                "description",
                "unknown conditions",
            )
        )

        city_name = data.get("name", city)

        return (
            f"In {city_name}, it is "
            f"{round(temperature)} degrees Fahrenheit "
            f"with {condition}. "
            f"It feels like {round(feels_like)} degrees. "
            f"Humidity is {humidity} percent, "
            f"wind speed is {wind_speed:.1f} miles per hour, "
            f"and cloud cover is {cloud_cover} percent."
        )

    def _normalize_forecast(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        city_data = data.get("city", {})
        entries = []

        for item in data.get("list", []):
            main = item.get("main", {})
            wind = item.get("wind", {})
            clouds = item.get("clouds", {})
            weather_items = item.get("weather") or [{}]

            rain = item.get("rain", {})
            snow = item.get("snow", {})

            condition = weather_items[0].get(
                "main",
                "Unknown",
            )

            description = weather_items[0].get(
                "description",
                "unknown conditions",
            )

            icon = weather_items[0].get(
                "icon",
                "",
            )

            entries.append(
                {
                    "timestamp": item.get("dt"),
                    "datetime": item.get("dt_txt"),
                    "temperature": main.get("temp"),
                    "feels_like": main.get(
                        "feels_like"
                    ),
                    "minimum": main.get("temp_min"),
                    "maximum": main.get("temp_max"),
                    "humidity": main.get("humidity"),
                    "pressure": main.get("pressure"),
                    "wind_speed": wind.get(
                        "speed",
                        0,
                    ),
                    "wind_direction": wind.get(
                        "deg",
                        0,
                    ),
                    "cloud_cover": clouds.get(
                        "all",
                        0,
                    ),
                    "precipitation_probability": (
                        float(item.get("pop", 0))
                        * 100
                    ),
                    "rain_inches": rain.get(
                        "3h",
                        0,
                    ),
                    "snow_inches": snow.get(
                        "3h",
                        0,
                    ),
                    "condition": condition,
                    "description": description,
                    "icon": icon,
                }
            )

        return {
            "city": city_data.get("name"),
            "country": city_data.get("country"),
            "timezone": city_data.get(
                "timezone",
                0,
            ),
            "entries": entries,
        }

    def _request(
        self,
        url: str,
        params: dict,
    ) -> dict[str, Any]:
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise WeatherToolError(
                f"Network request failed: {error}"
            ) from error

        if response.status_code != 200:
            raise WeatherToolError(
                f"OpenWeather returned status "
                f"{response.status_code}: "
                f"{response.text}"
            )

        return response.json()

    def _validate_key(self) -> None:
        if not self.api_key:
            raise WeatherToolError(
                "OPENWEATHER_API_KEY is missing."
            )