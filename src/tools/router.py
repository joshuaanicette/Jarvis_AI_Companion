import re
import time
import webbrowser
from urllib.parse import quote


class ToolRouter:
    def __init__(
        self,
        registry,
        app=None,
    ):
        self.registry = registry
        self.app = app

    def check_tools(
        self,
        text: str,
    ) -> str | None:
        """
        Check whether a request should be handled by a tool.

        Returns:
            A response string when a tool handles the request.
            None when the request should go to the language model.
        """

        normalized = self._normalize(text)

        # Cancel active background work without shutting down Jay.
        if self._matches_any(
            normalized,
            {
                "cancel task",
                "cancel the task",
                "cancel current task",
                "cancel the current task",
                "stop current task",
                "stop the current task",
            },
        ):
            if self.app is None:
                return (
                    "Application task control is not available."
                )

            return self.app.cancel_active_task()

        # Stop the camera.
        if self._matches_any(
            normalized,
            {
                "stop camera",
                "stop the camera",
                "turn off camera",
                "turn off the camera",
                "close camera",
                "close the camera",
                "disconnect camera",
                "disconnect the camera",
                "stop object detection",
                "stop detecting objects",
            },
        ):
            return self._stop_camera()

        # Start the camera and continuous object detection.
        if self._is_continuous_camera_request(normalized):
            return self._start_continuous_camera()

        # Perform a one-time room scan.
        if self._is_room_scan_request(normalized):
            return self._scan_room_once()

        # Ask what the camera currently sees.
        if self._is_scene_request(normalized):
            return self._get_scene_summary()

        # Ask about the largest visible object.
        if self._is_largest_object_request(normalized):
            return self._get_largest_object()

        # Ask for dimensions, area, or distance.
        object_label = (
            self._extract_object_measurement_request(
                normalized
            )
        )

        if object_label:
            return self._describe_object(
                object_label
            )

        # Clothing recommendation.
        if self._is_clothing_request(normalized):
            return self._get_clothing_advice(
                text
            )

        # Open the dashboard only when dashboard is requested.
        if self._is_dashboard_request(normalized):
            return self._open_weather_dashboard(
                text
            )

        # Normal spoken weather request.
        if self._is_weather_request(normalized):
            city = self._extract_city(text)

            return self._run_registered_tool(
                "weather",
                city,
            )

        # Time request.
        if self._is_time_request(normalized):
            return self._run_registered_tool(
                "time",
                text,
            )

        # Allow other registered tools to handle the request.
        return self._check_registered_tools(
            text
        )

    def route(
        self,
        text: str,
    ) -> str | None:
        """
        Compatibility alias for older code.
        """

        return self.check_tools(text)

    def _check_registered_tools(
        self,
        text: str,
    ) -> str | None:
        tools = self._get_registered_tools()

        for tool in tools:
            can_handle = getattr(
                tool,
                "can_handle",
                None,
            )

            if not callable(can_handle):
                continue

            try:
                if not can_handle(text):
                    continue

                return self._execute_tool(
                    tool,
                    text,
                )

            except Exception:
                continue

        return None

    def _run_registered_tool(
        self,
        tool_name: str,
        text: str,
    ) -> str | None:
        tool = self._find_tool(
            tool_name
        )

        if tool is None:
            return (
                f"The {tool_name} tool is not registered."
            )

        try:
            return self._execute_tool(
                tool,
                text,
            )

        except Exception as error:
            return (
                f"The {tool_name} tool failed: {error}"
            )

    @staticmethod
    def _execute_tool(
        tool,
        text: str,
    ) -> str:
        method_names = (
            "run",
            "execute",
            "handle",
            "invoke",
        )

        for method_name in method_names:
            method = getattr(
                tool,
                method_name,
                None,
            )

            if not callable(method):
                continue

            result = method(text)

            if result is None:
                return ""

            return str(result)

        raise AttributeError(
            f"{tool.__class__.__name__} does not provide "
            "run(), execute(), handle(), or invoke()."
        )

    def _get_registered_tools(
        self,
    ) -> list:
        possible_attributes = (
            "tools",
            "_tools",
            "registry",
            "_registry",
        )

        for attribute_name in possible_attributes:
            value = getattr(
                self.registry,
                attribute_name,
                None,
            )

            if isinstance(value, dict):
                return list(
                    value.values()
                )

            if isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                return list(value)

        list_tools = getattr(
            self.registry,
            "list_tools",
            None,
        )

        if callable(list_tools):
            result = list_tools()

            if isinstance(result, dict):
                return list(
                    result.values()
                )

            return list(result)

        return []

    def _find_tool(
        self,
        requested_name: str,
    ):
        normalized_requested = (
            requested_name.lower()
        )

        registry_get = getattr(
            self.registry,
            "get",
            None,
        )

        if callable(registry_get):
            try:
                tool = registry_get(
                    requested_name
                )

                if tool is not None:
                    return tool

            except Exception:
                pass

        for tool in self._get_registered_tools():
            possible_names = [
                getattr(
                    tool,
                    "name",
                    "",
                ),
                tool.__class__.__name__,
            ]

            for possible_name in possible_names:
                normalized_name = (
                    str(possible_name)
                    .lower()
                    .replace("tool", "")
                    .strip()
                )

                if normalized_name == normalized_requested:
                    return tool

        return None

    def _start_continuous_camera(
        self,
    ) -> str:
        if self.app is None:
            return (
                "Camera control is not available."
            )

        if self.app.vision.running:
            return (
                "The camera and object detection "
                "are already running."
            )

        try:
            started = (
                self.app.vision.start_continuous(
                    show_camera=True,
                    detection_interval=0.1,
                )
            )

        except Exception as error:
            return (
                "I could not connect to the camera. "
                f"The camera reported: {error}"
            )

        if started:
            return (
                "I connected to the camera and started "
                "continuous object detection."
            )

        return (
            "I could not start the camera. "
            "It may already be in use by another program."
        )

    def _stop_camera(
        self,
    ) -> str:
        if self.app is None:
            return (
                "Camera control is not available."
            )

        if self.app.vision.running:
            self.app.vision.stop_continuous()

            return (
                "I stopped the camera and "
                "continuous object detection."
            )

        if self.app.vision.started:
            self.app.vision.stop()

            return (
                "I stopped the camera."
            )

        return (
            "The camera is not currently running."
        )

    def _scan_room_once(
        self,
    ) -> str:
        if self.app is None:
            return (
                "Room scanning is not available."
            )

        try:
            self.app.vision.process_once(
                show_camera=True,
                display_seconds=3.0,
            )

            summary = (
                self.app.vision.get_room_summary()
            )

            if not self.app.vision.running:
                self.app.vision.stop()

            return summary

        except Exception as error:
            return (
                f"I could not scan the room: {error}"
            )

    def _get_scene_summary(
        self,
    ) -> str:
        if self.app is None:
            return (
                "Vision is not available."
            )

        if self.app.vision.latest_scan is not None:
            return (
                self.app.vision.get_room_summary()
            )

        if self.app.vision.latest_scene:
            return self.app.vision.latest_scene

        return (
            "I do not have a recent camera scan. "
            "Ask me to scan the room or start "
            "the camera first."
        )

    def _describe_object(
        self,
        object_label: str,
    ) -> str:
        if self.app is None:
            return (
                "Object measurement is not available."
            )

        if self.app.vision.latest_scan is None:
            return (
                "I do not have a recent room scan. "
                "Ask me to scan the room or start "
                "the camera first."
            )

        return self.app.vision.describe_object(
            object_label
        )

    def _get_largest_object(
        self,
    ) -> str:
        if self.app is None:
            return (
                "Object measurement is not available."
            )

        if self.app.vision.latest_scan is None:
            return (
                "I do not have a recent room scan."
            )

        return (
            self.app.vision.get_largest_object()
        )

    def _get_clothing_advice(
        self,
        text: str,
    ) -> str:
        if self.app is None:
            return (
                "Clothing advice is not available."
            )

        city = self._extract_city(text)

        try:
            weather_data = (
                self.app.weather_tool.get_weather(
                    city
                )
            )

        except AttributeError:
            weather_response = (
                self._run_registered_tool(
                    "weather",
                    city,
                )
            )

            return (
                weather_response
                or "I could not retrieve the weather."
            )

        advisor = self.app.clothing_advisor

        method_names = (
            "recommend",
            "advise",
            "get_advice",
        )

        for method_name in method_names:
            method = getattr(
                advisor,
                method_name,
                None,
            )

            if callable(method):
                return str(
                    method(weather_data)
                )

        return (
            "The clothing advisor does not have "
            "a supported recommendation method."
        )

    def _open_weather_dashboard(
        self,
        text: str,
    ) -> str:
        """
        Start the local weather dashboard and automatically
        open it in a new browser tab.
        """

        if self.app is None:
            return (
                "The weather dashboard is not available."
            )

        city = self._extract_city(text)

        url = (
            "http://127.0.0.1:8765/"
            f"?city={quote(city)}"
        )

        try:
            self.app.weather_dashboard.start()

            # Give the local FastAPI server time to start.
            time.sleep(1.0)

            opened = webbrowser.open_new_tab(
                url
            )

            if opened:
                return (
                    f"I opened the weather dashboard "
                    f"for {city} in your browser."
                )

            return (
                "The weather dashboard is running, "
                "but I could not open the browser "
                f"automatically. Open {url}"
            )

        except Exception as error:
            return (
                "I could not open the weather dashboard. "
                f"The error was: {error}"
            )

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        normalized = text.lower().strip()

        normalized = re.sub(
            r"[^\w\s'-]",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _matches_any(
        text: str,
        phrases: set[str],
    ) -> bool:
        return text in phrases

    @staticmethod
    def _is_time_request(
        text: str,
    ) -> bool:
        phrases = (
            "what time",
            "current time",
            "tell me the time",
            "time is it",
        )

        return any(
            phrase in text
            for phrase in phrases
        )

    @staticmethod
    def _is_weather_request(
        text: str,
    ) -> bool:
        phrases = (
            "weather",
            "temperature",
            "forecast",
            "rain today",
            "snow today",
        )

        return any(
            phrase in text
            for phrase in phrases
        )

    @staticmethod
    def _is_dashboard_request(
        text: str,
    ) -> bool:
        dashboard_phrases = (
            "weather dashboard",
            "show weather dashboard",
            "show me the weather dashboard",
            "open weather dashboard",
            "open the weather dashboard",
            "display weather dashboard",
            "display the weather dashboard",
            "launch weather dashboard",
            "launch the weather dashboard",
        )

        return any(
            phrase in text
            for phrase in dashboard_phrases
        )

    @staticmethod
    def _is_clothing_request(
        text: str,
    ) -> bool:
        phrases = (
            "what should i wear",
            "what should i put on",
            "clothing recommendation",
            "clothes should i wear",
            "dress for the weather",
        )

        return any(
            phrase in text
            for phrase in phrases
        )

    @staticmethod
    def _is_continuous_camera_request(
        text: str,
    ) -> bool:
        exact_phrases = (
            "start camera",
            "start the camera",
            "open camera",
            "open the camera",
            "turn on camera",
            "turn on the camera",
            "connect to camera",
            "connect to the camera",
            "activate camera",
            "activate the camera",
            "enable camera",
            "enable the camera",
            "start object detection",
            "start detecting objects",
            "turn on object detection",
            "begin object detection",
            "keep detecting objects",
            "scan continuously",
        )

        if any(
            phrase in text
            for phrase in exact_phrases
        ):
            return True

        camera_words = (
            "camera",
            "object detection",
            "detect objects",
        )

        start_words = (
            "start",
            "open",
            "connect",
            "activate",
            "turn on",
            "enable",
            "begin",
        )

        return (
            any(
                word in text
                for word in camera_words
            )
            and any(
                word in text
                for word in start_words
            )
        )

    @staticmethod
    def _is_room_scan_request(
        text: str,
    ) -> bool:
        phrases = (
            "scan the room",
            "scan room",
            "look around the room",
            "identify objects in the room",
            "detect objects in the room",
        )

        return any(
            phrase in text
            for phrase in phrases
        )

    @staticmethod
    def _is_scene_request(
        text: str,
    ) -> bool:
        phrases = (
            "what do you see",
            "what are you seeing",
            "what are you detecting",
            "what objects do you see",
            "describe the room",
            "describe what you see",
        )

        return any(
            phrase in text
            for phrase in phrases
        )

    @staticmethod
    def _is_largest_object_request(
        text: str,
    ) -> bool:
        phrases = (
            "largest object",
            "biggest object",
            "what takes up the most space",
        )

        return any(
            phrase in text
            for phrase in phrases
        )

    @staticmethod
    def _extract_object_measurement_request(
        text: str,
    ) -> str | None:
        patterns = (
            (
                r"(?:dimensions|size|measurements) of "
                r"(?:the |a |an )?(.+)$"
            ),
            (
                r"how (?:big|wide|tall|large) is "
                r"(?:the |a |an )?(.+)$"
            ),
            (
                r"how far (?:away )?is "
                r"(?:the |a |an )?(.+)$"
            ),
            (
                r"(?:area|detected area) of "
                r"(?:the |a |an )?(.+)$"
            ),
            (
                r"measure "
                r"(?:the |a |an )?(.+)$"
            ),
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
            )

            if not match:
                continue

            label = match.group(1).strip()

            label = re.sub(
                r"\b(?:object|please|for me)\b$",
                "",
                label,
            ).strip()

            if label:
                return label

        return None

    @staticmethod
    def _extract_city(
        text: str,
    ) -> str:
        cleaned = text.strip()

        cleaned = re.sub(
            r"[?!.,]+$",
            "",
            cleaned,
        ).strip()

        patterns = (
            r"\bweather\s+dashboard\s+for\s+([a-zA-Z .'-]+)$",
            r"\bweather\s+dashboard\s+in\s+([a-zA-Z .'-]+)$",
            r"\bdashboard\s+for\s+([a-zA-Z .'-]+)$",
            r"\bdashboard\s+in\s+([a-zA-Z .'-]+)$",
            r"\bweather\s+in\s+([a-zA-Z .'-]+)$",
            r"\bforecast\s+in\s+([a-zA-Z .'-]+)$",
            r"\btemperature\s+in\s+([a-zA-Z .'-]+)$",
            r"\bweather\s+for\s+([a-zA-Z .'-]+)$",
            r"\bforecast\s+for\s+([a-zA-Z .'-]+)$",
            r"\bin\s+([a-zA-Z .'-]+)$",
            r"\bfor\s+([a-zA-Z .'-]+)$",
        )

        ignored_words = {
            "today",
            "tomorrow",
            "outside",
            "right now",
            "currently",
        }

        for pattern in patterns:
            match = re.search(
                pattern,
                cleaned,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            city = match.group(1).strip()

            city = re.sub(
                r"\b(today|tomorrow|right now|currently)\b$",
                "",
                city,
                flags=re.IGNORECASE,
            ).strip()

            if (
                city
                and city.lower() not in ignored_words
            ):
                return city.title()

        return "Philadelphia"