from typing import Any


class ClothingAdvisor:
    """
    Produces practical clothing recommendations from weather data.
    """

    def recommend(
        self,
        weather: dict[str, Any],
    ) -> str:
        main = weather.get("main", {})
        wind = weather.get("wind", {})
        clouds = weather.get("clouds", {})
        weather_items = weather.get("weather") or [{}]

        temperature = float(
            main.get("temp", 70)
        )

        feels_like = float(
            main.get(
                "feels_like",
                temperature,
            )
        )

        humidity = int(
            main.get("humidity", 0)
        )

        wind_speed = float(
            wind.get("speed", 0)
        )

        cloud_cover = int(
            clouds.get("all", 0)
        )

        condition = str(
            weather_items[0].get(
                "main",
                "",
            )
        ).lower()

        description = str(
            weather_items[0].get(
                "description",
                "",
            )
        ).lower()

        effective_temperature = feels_like
        recommendations = []

        if effective_temperature >= 90:
            recommendations.extend(
                [
                    "wear lightweight and breathable clothing",
                    "choose shorts and a light short-sleeve shirt",
                    "wear sunscreen and bring water",
                ]
            )

        elif effective_temperature >= 80:
            recommendations.extend(
                [
                    "wear lightweight clothing",
                    "shorts and a T-shirt should be comfortable",
                ]
            )

        elif effective_temperature >= 70:
            recommendations.extend(
                [
                    "wear a T-shirt with jeans, shorts, or light pants",
                ]
            )

        elif effective_temperature >= 60:
            recommendations.extend(
                [
                    "wear long pants and a light jacket or sweatshirt",
                ]
            )

        elif effective_temperature >= 50:
            recommendations.extend(
                [
                    "wear long pants, a sweater, and a light-to-medium jacket",
                ]
            )

        elif effective_temperature >= 40:
            recommendations.extend(
                [
                    "wear a warm jacket, long pants, and closed shoes",
                ]
            )

        elif effective_temperature >= 32:
            recommendations.extend(
                [
                    "wear a winter coat, long pants, and warm layers",
                    "consider gloves and a hat",
                ]
            )

        else:
            recommendations.extend(
                [
                    "wear a heavy winter coat and thermal layers",
                    "wear gloves, a hat, and insulated shoes",
                ]
            )

        rainy = any(
            word in condition or word in description
            for word in (
                "rain",
                "drizzle",
                "thunderstorm",
            )
        )

        snowy = any(
            word in condition or word in description
            for word in (
                "snow",
                "sleet",
                "ice",
            )
        )

        sunny = (
            "clear" in condition
            or "sun" in description
        )

        if rainy:
            recommendations.append(
                "bring an umbrella or wear a waterproof jacket"
            )

        if snowy:
            recommendations.append(
                "wear waterproof boots with good traction"
            )

        if sunny and temperature >= 65:
            recommendations.append(
                "consider sunglasses"
            )

        if wind_speed >= 20:
            recommendations.append(
                "wear a wind-resistant outer layer"
            )

        elif wind_speed >= 12:
            recommendations.append(
                "a light windbreaker may help"
            )

        if humidity >= 75 and temperature >= 75:
            recommendations.append(
                "choose moisture-wicking or breathable fabric"
            )

        if cloud_cover >= 85 and temperature < 60:
            recommendations.append(
                "the heavy cloud cover may make it feel cooler"
            )

        city = weather.get(
            "name",
            "your location",
        )

        return (
            f"For {city}, I recommend that you "
            + ", ".join(recommendations)
            + "."
        )