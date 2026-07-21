from __future__ import annotations

import re

from src.tools.tool import Tool


class NavigationTool(Tool):
    def __init__(
        self,
        dashboard,
        location_manager,
    ):
        self.dashboard = dashboard
        self.location_manager = location_manager

    @property
    def name(self):
        return "navigation"

    def can_handle(
        self,
        text: str,
    ) -> bool:
        normalized = self._normalize(text)

        return any(
            phrase in normalized
            for phrase in (
                "how long does it take to",
                "how long to",
                "how far to",
                "compare routes to",
                "drive to",
                "walk to",
                "train to",
                "transit to",
                "navigation dashboard",
                "open navigation",
                "show the map",
            )
        )

    def execute(
        self,
        text: str = "",
    ) -> str:
        normalized = self._normalize(text)

        if any(
            phrase in normalized
            for phrase in (
                "navigation dashboard",
                "open navigation",
                "show the map",
            )
        ):
            return self.dashboard.open()

        destination = self._extract_destination(
            text
        )

        if not destination:
            return (
                "Tell me the destination after the word to. "
                "For example: Jay, how long does it take "
                "to Carnegie Mellon University?"
            )

        try:
            comparison = (
                self.location_manager
                .compare_home_to_destination(
                    destination_query=destination
                )
            )
        except Exception as error:
            return (
                "I could not compare those routes: "
                f"{error}"
            )

        lines = [
            (
                f"From {comparison.start.label} to "
                f"{comparison.destination.label}:"
            ),
            "",
        ]

        labels = {
            "driving": "Driving",
            "walking": "Walking",
            "transit": "Train or transit",
        }

        for mode in (
            "driving",
            "walking",
            "transit",
        ):
            route = comparison.routes.get(mode)

            if route is None or not route.available:
                reason = (
                    route.error
                    if route is not None
                    else "No route was returned."
                )

                lines.append(
                    f"{labels[mode]}: unavailable ({reason})"
                )
                continue

            lines.append(
                f"{labels[mode]}: "
                f"{self._format_duration(route.duration_minutes)}, "
                f"{route.distance_miles:.1f} miles"
            )

        return "\n".join(lines)

    def run(
        self,
        text: str = "",
    ) -> str:
        return self.execute(text)

    @staticmethod
    def _extract_destination(
        text: str,
    ) -> str:
        match = re.search(
            r"\bto\s+(.+)$",
            str(text).strip(),
            flags=re.IGNORECASE,
        )

        if match is None:
            return ""

        return match.group(1).strip(
            " .?!"
        )

    @staticmethod
    def _format_duration(
        minutes: float,
    ) -> str:
        total_minutes = max(
            0,
            round(minutes),
        )

        if total_minutes < 60:
            return f"{total_minutes} minutes"

        hours, remaining_minutes = divmod(
            total_minutes,
            60,
        )

        hour_word = (
            "hour"
            if hours == 1
            else "hours"
        )

        if remaining_minutes == 0:
            return f"{hours} {hour_word}"

        return (
            f"{hours} {hour_word} "
            f"{remaining_minutes} minutes"
        )

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(text).lower().strip(),
        )