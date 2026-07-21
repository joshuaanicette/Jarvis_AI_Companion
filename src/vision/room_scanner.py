from __future__ import annotations

import math
from typing import Any

from src.core.logger import logger


class RoomScanner:
    """
    Analyze detections and estimate object size and distance.

    The scanner accepts detection dictionaries produced by
    ObjectDetector.

    Distance estimation uses:

        distance =
            focal_length_pixels
            * known_object_height
            / detected_height_pixels

    Constructor compatibility:
    - frame_width
    - frame_height
    - camera_width_pixels
    - camera_height_pixels
    - horizontal_fov
    - horizontal_fov_degrees
    - person_height
    - person_height_meters
    - minimum_confidence
    - focal_length_pixels
    """

    DEFAULT_OBJECT_HEIGHTS_METERS = {
        "person": 1.70,
        "chair": 0.90,
        "bottle": 0.24,
        "cup": 0.10,
        "cell phone": 0.15,
        "laptop": 0.24,
        "tv": 0.60,
        "book": 0.24,
        "backpack": 0.45,
        "dog": 0.60,
        "cat": 0.25,
    }

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        focal_length_x_pixels: float = 700.0,
        focal_length_y_pixels: float = 700.0,
        minimum_confidence: float = 0.45,
        person_height_meters: float = 1.70,
        **kwargs,
    ):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)

        self.camera_width_pixels = self.frame_width
        self.camera_height_pixels = self.frame_height

        self.horizontal_fov_degrees = float(
            kwargs.get(
                "horizontal_fov_degrees",
                kwargs.get(
                    "horizontal_fov",
                    55.0,
                ),
            )
        )

        self.focal_length_x_pixels = float(
            focal_length_x_pixels
        )

        self.focal_length_y_pixels = float(
        focal_length_y_pixels
        )

    # Older methods may use this general focal-length name.
        self.focal_length_pixels = (
            self.focal_length_y_pixels
        )

        self.minimum_confidence = float(
            minimum_confidence
        )

        self.person_height_meters = float(
            person_height_meters
        )

        self.object_heights_meters = {
            "person": self.person_height_meters,
            "chair": 0.90,
            "bottle": 0.24,
            "cup": 0.10,
            "cell phone": 0.15,
            "laptop": 0.24,
            "tv": 0.60,
            "book": 0.24,
            "backpack": 0.45,
            "dog": 0.60,
            "cat": 0.25,
        }

        self.latest_scan = None
        self.extra_options = dict(kwargs)

        logger.info(
            "Room scanner initialized: "
            "%sx%s, focal lengths %.1f x %.1f pixels",
            self.frame_width,
            self.frame_height,
            self.focal_length_x_pixels,
            self.focal_length_y_pixels,
        )

    def scan(
        self,
        frame=None,
        detections: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a frame and its detections.

        Supported calls:

            scan(frame=frame, detections=detections)
            scan(frame, detections)
            scan(detections)
        """

        if detections is None:
            if isinstance(frame, list):
                detections = frame
                frame = None
            else:
                detections = []

        frame_width = self.frame_width
        frame_height = self.frame_height

        frame_shape = getattr(
            frame,
            "shape",
            None,
        )

        if (
            frame_shape is not None
            and len(frame_shape) >= 2
        ):
            frame_height = int(
                frame_shape[0]
            )

            frame_width = int(
                frame_shape[1]
            )

        measured_objects: list[dict[str, Any]] = []

        for detection in detections:
            confidence = self._get_confidence(
                detection
            )

            if (
                confidence is not None
                and confidence < self.minimum_confidence
            ):
                continue

            measurement = self._measure_detection(
                detection=detection,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            if measurement is not None:
                measured_objects.append(
                    measurement
                )

        summary = self._build_summary(
            measured_objects
        )

        result = {
            "summary": summary,
            "description": summary,
            "scene": summary,
            "objects": measured_objects,
            "object_count": len(
                measured_objects
            ),
            "frame_width_pixels": frame_width,
            "frame_height_pixels": frame_height,
            "horizontal_fov_degrees": (
                self.horizontal_fov_degrees
            ),
            "focal_length_pixels": (
                self.focal_length_pixels
            ),
            "minimum_confidence": (
                self.minimum_confidence
            ),
        }

        self.latest_scan = result

        return result

    def analyze(
        self,
        frame=None,
        detections: list[Any] | None = None,
    ) -> dict[str, Any]:
        return self.scan(
            frame=frame,
            detections=detections,
        )

    def process(
        self,
        frame=None,
        detections: list[Any] | None = None,
    ) -> dict[str, Any]:
        return self.scan(
            frame=frame,
            detections=detections,
        )

    def describe_object(
        self,
        scan_result=None,
        object_label: str | None = None,
    ) -> str:
        """
        Describe an object from the latest scan.

        Supported calls:

            describe_object(scan, "person")
            describe_object("person", scan)
            describe_object("person")
        """

        scan: dict[str, Any] | None
        requested = ""

        if isinstance(scan_result, dict):
            scan = scan_result

            if isinstance(object_label, str):
                requested = object_label

        elif isinstance(scan_result, str):
            requested = scan_result

            if isinstance(object_label, dict):
                scan = object_label
            else:
                scan = self.latest_scan

        elif isinstance(object_label, str):
            requested = object_label
            scan = self.latest_scan

        else:
            scan = self.latest_scan

        requested = requested.strip().lower()

        if not requested:
            return (
                "Please specify which object you want me "
                "to describe."
            )

        if not isinstance(scan, dict):
            return (
                "I do not have a recent room scan."
            )

        objects = scan.get(
            "objects",
            [],
        )

        matching = []

        for item in objects:
            if not isinstance(item, dict):
                continue

            label = str(
                item.get(
                    "label",
                    "",
                )
            ).lower()

            if (
                requested == label
                or requested in label
                or label in requested
            ):
                matching.append(item)

        if not matching:
            return (
                f"I did not find a {requested} "
                "in the latest scan."
            )

        selected = max(
            matching,
            key=lambda item: float(
                item.get(
                    "area_pixels",
                    0.0,
                )
            ),
        )

        return self._format_object_description(
            selected
        )

    def get_object_description(
        self,
        scan_result=None,
        object_label: str | None = None,
    ) -> str:
        return self.describe_object(
            scan_result,
            object_label,
        )

    def measure_object(
        self,
        scan_result=None,
        object_label: str | None = None,
    ) -> str:
        return self.describe_object(
            scan_result,
            object_label,
        )

    def calibrate_focal_length(
        self,
        known_distance_meters: float,
        known_height_meters: float,
        detected_height_pixels: float,
    ) -> float:
        known_distance = float(
            known_distance_meters
        )

        known_height = float(
            known_height_meters
        )

        detected_height = float(
            detected_height_pixels
        )

        if known_distance <= 0:
            raise ValueError(
                "Known distance must be greater than zero."
            )

        if known_height <= 0:
            raise ValueError(
                "Known height must be greater than zero."
            )

        if detected_height <= 0:
            raise ValueError(
                "Detected pixel height must be greater than zero."
            )

        focal_length = (
            detected_height
            * known_distance
            / known_height
        )

        self.focal_length_pixels = focal_length

        logger.info(
            "Camera focal length calibrated to %.2f pixels",
            focal_length,
        )

        return focal_length

    def set_known_object_height(
        self,
        label: str,
        height_meters: float,
    ) -> None:
        normalized_label = (
            label.strip().lower()
        )

        height = float(
            height_meters
        )

        if not normalized_label:
            raise ValueError(
                "Object label cannot be empty."
            )

        if height <= 0:
            raise ValueError(
                "Object height must be greater than zero."
            )

        self.object_heights_meters[
            normalized_label
        ] = height

    def _measure_detection(
        self,
        detection: Any,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any] | None:
        label = self._get_label(
            detection
        )

        confidence = self._get_confidence(
            detection
        )

        bbox = self._get_bbox(
            detection
        )

        if bbox is None:
            return {
                "label": label,
                "confidence": confidence,
                "bbox": None,
                "width_pixels": None,
                "height_pixels": None,
                "area_pixels": 0.0,
                "distance_meters": None,
                "distance_feet": None,
                "estimated_width_meters": None,
                "estimated_height_meters": None,
                "horizontal_position": "unknown",
                "vertical_position": "unknown",
            }

        x1, y1, x2, y2 = bbox

        x1 = self._clamp(
            x1,
            0.0,
            float(frame_width),
        )

        x2 = self._clamp(
            x2,
            0.0,
            float(frame_width),
        )

        y1 = self._clamp(
            y1,
            0.0,
            float(frame_height),
        )

        y2 = self._clamp(
            y2,
            0.0,
            float(frame_height),
        )

        width_pixels = max(
            0.0,
            x2 - x1,
        )

        height_pixels = max(
            0.0,
            y2 - y1,
        )

        area_pixels = (
            width_pixels
            * height_pixels
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        known_height = self._known_height_for_label(
            label
        )

        distance_meters = None
        distance_feet = None
        estimated_width_meters = None
        estimated_height_meters = None

        if (
            known_height is not None
            and height_pixels > 0
            and self.focal_length_pixels > 0
        ):
            distance_meters = (
                self.focal_length_pixels
                * known_height
                / height_pixels
            )

            distance_feet = (
                distance_meters
                * 3.28084
            )

            estimated_width_meters = (
                width_pixels
                * distance_meters
                / self.focal_length_pixels
            )

            estimated_height_meters = (
                height_pixels
                * distance_meters
                / self.focal_length_pixels
            )

        horizontal_angle = self._estimate_horizontal_angle(
            center_x=center_x,
            frame_width=frame_width,
        )

        return {
            "label": label,
            "confidence": confidence,
            "bbox": [
                x1,
                y1,
                x2,
                y2,
            ],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y,
            "width_pixels": width_pixels,
            "height_pixels": height_pixels,
            "area_pixels": area_pixels,
            "known_height_meters": known_height,
            "distance_meters": distance_meters,
            "distance_feet": distance_feet,
            "estimated_width_meters": (
                estimated_width_meters
            ),
            "estimated_height_meters": (
                estimated_height_meters
            ),
            "horizontal_position": (
                self._horizontal_position(
                    center_x=center_x,
                    frame_width=frame_width,
                )
            ),
            "vertical_position": (
                self._vertical_position(
                    center_y=center_y,
                    frame_height=frame_height,
                )
            ),
            "horizontal_angle_degrees": (
                horizontal_angle
            ),
        }

    def _build_summary(
        self,
        objects: list[dict[str, Any]],
    ) -> str:
        if not objects:
            return (
                "I opened the camera, but I did not "
                "recognize any objects."
            )

        people = [
            item
            for item in objects
            if str(
                item.get(
                    "label",
                    "",
                )
            ).lower() == "person"
        ]

        if people:
            person = max(
                people,
                key=lambda item: float(
                    item.get(
                        "area_pixels",
                        0.0,
                    )
                ),
            )

            return self._format_person_summary(
                person=person,
                total_objects=len(objects),
            )

        labels = [
            str(
                item.get(
                    "label",
                    "object",
                )
            )
            for item in objects
        ]

        return (
            "I can see "
            + self._join_labels(labels)
            + "."
        )

    def _format_person_summary(
        self,
        person: dict[str, Any],
        total_objects: int,
    ) -> str:
        parts = [
            "I can see a person"
        ]

        confidence = person.get(
            "confidence"
        )

        width_pixels = person.get(
            "width_pixels"
        )

        height_pixels = person.get(
            "height_pixels"
        )

        distance_meters = person.get(
            "distance_meters"
        )

        distance_feet = person.get(
            "distance_feet"
        )

        estimated_width = person.get(
            "estimated_width_meters"
        )

        estimated_height = person.get(
            "estimated_height_meters"
        )

        position = person.get(
            "horizontal_position"
        )

        if confidence is not None:
            parts.append(
                f"with {float(confidence):.0%} confidence"
            )

        if position and position != "unknown":
            parts.append(
                f"near the {position} of the camera view"
            )

        if (
            width_pixels is not None
            and height_pixels is not None
        ):
            parts.append(
                f"occupying approximately "
                f"{float(width_pixels):.0f} pixels wide "
                f"and {float(height_pixels):.0f} pixels tall"
            )

        if (
            estimated_width is not None
            and estimated_height is not None
        ):
            parts.append(
                f"with an estimated visible width of "
                f"{float(estimated_width):.2f} meters "
                f"and height of "
                f"{float(estimated_height):.2f} meters"
            )

        if (
            distance_meters is not None
            and distance_feet is not None
        ):
            parts.append(
                f"approximately "
                f"{float(distance_meters):.2f} meters, "
                f"or {float(distance_feet):.1f} feet, "
                "from the camera"
            )

        if total_objects > 1:
            other_count = (
                total_objects - 1
            )

            parts.append(
                f"along with {other_count} other "
                f"detected object"
                + (
                    ""
                    if other_count == 1
                    else "s"
                )
            )

        return (
            ", ".join(parts)
            + "."
        )

    def _format_object_description(
        self,
        item: dict[str, Any],
    ) -> str:
        label = str(
            item.get(
                "label",
                "object",
            )
        )

        parts = [
            f"I detected a {label}"
        ]

        confidence = item.get(
            "confidence"
        )

        width_pixels = item.get(
            "width_pixels"
        )

        height_pixels = item.get(
            "height_pixels"
        )

        distance_meters = item.get(
            "distance_meters"
        )

        distance_feet = item.get(
            "distance_feet"
        )

        estimated_width = item.get(
            "estimated_width_meters"
        )

        estimated_height = item.get(
            "estimated_height_meters"
        )

        if confidence is not None:
            parts.append(
                f"with {float(confidence):.0%} confidence"
            )

        if (
            width_pixels is not None
            and height_pixels is not None
        ):
            parts.append(
                f"measuring approximately "
                f"{float(width_pixels):.0f} pixels wide "
                f"and {float(height_pixels):.0f} pixels tall "
                "in the image"
            )

        if (
            estimated_width is not None
            and estimated_height is not None
        ):
            parts.append(
                f"with an estimated visible width of "
                f"{float(estimated_width):.2f} meters "
                f"and height of "
                f"{float(estimated_height):.2f} meters"
            )

        if (
            distance_meters is not None
            and distance_feet is not None
        ):
            parts.append(
                f"approximately "
                f"{float(distance_meters):.2f} meters, "
                f"or {float(distance_feet):.1f} feet, "
                "from the camera"
            )
        else:
            parts.append(
                "but its real-world distance cannot be estimated "
                "without a known reference height"
            )

        return (
            ", ".join(parts)
            + "."
        )

    @staticmethod
    def _calculate_focal_length_from_fov(
        frame_width: int,
        horizontal_fov_degrees: float,
    ) -> float:
        width = float(
            frame_width
        )

        fov = float(
            horizontal_fov_degrees
        )

        if width <= 0:
            raise ValueError(
                "Frame width must be greater than zero."
            )

        if not 1.0 < fov < 179.0:
            raise ValueError(
                "Horizontal field of view must be "
                "between 1 and 179 degrees."
            )

        half_fov_radians = math.radians(
            fov / 2.0
        )

        return (
            width
            / (
                2.0
                * math.tan(
                    half_fov_radians
                )
            )
        )

    def _estimate_horizontal_angle(
        self,
        center_x: float,
        frame_width: int,
    ) -> float:
        if frame_width <= 0:
            return 0.0

        normalized_offset = (
            center_x
            - (
                frame_width / 2.0
            )
        ) / (
            frame_width / 2.0
        )

        return (
            normalized_offset
            * (
                self.horizontal_fov_degrees
                / 2.0
            )
        )

    def _known_height_for_label(
        self,
        label: str,
    ) -> float | None:
        normalized = (
            label.strip().lower()
        )

        return self.object_heights_meters.get(
            normalized
        )

    @staticmethod
    def _get_label(
        detection: Any,
    ) -> str:
        if isinstance(detection, dict):
            for key in (
                "label",
                "name",
                "class_name",
                "class",
            ):
                value = detection.get(key)

                if value is not None:
                    return str(value)

        for attribute in (
            "label",
            "name",
            "class_name",
        ):
            value = getattr(
                detection,
                attribute,
                None,
            )

            if value is not None:
                return str(value)

        return "object"

    @staticmethod
    def _get_confidence(
        detection: Any,
    ) -> float | None:
        value = None

        if isinstance(detection, dict):
            for key in (
                "confidence",
                "score",
                "conf",
            ):
                if key in detection:
                    value = detection[key]
                    break
        else:
            for attribute in (
                "confidence",
                "score",
                "conf",
            ):
                value = getattr(
                    detection,
                    attribute,
                    None,
                )

                if value is not None:
                    break

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_bbox(
        detection: Any,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ] | None:
        bbox = None

        if isinstance(detection, dict):
            bbox = (
                detection.get("bbox")
                or detection.get("box")
                or detection.get("bounding_box")
            )

            if bbox is None and all(
                key in detection
                for key in (
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                )
            ):
                bbox = (
                    detection["x1"],
                    detection["y1"],
                    detection["x2"],
                    detection["y2"],
                )
        else:
            bbox = (
                getattr(
                    detection,
                    "bbox",
                    None,
                )
                or getattr(
                    detection,
                    "box",
                    None,
                )
                or getattr(
                    detection,
                    "bounding_box",
                    None,
                )
            )

        if isinstance(bbox, dict):
            values = (
                bbox.get(
                    "x1",
                    bbox.get("left"),
                ),
                bbox.get(
                    "y1",
                    bbox.get("top"),
                ),
                bbox.get(
                    "x2",
                    bbox.get("right"),
                ),
                bbox.get(
                    "y2",
                    bbox.get("bottom"),
                ),
            )

        elif (
            isinstance(
                bbox,
                (list, tuple),
            )
            and len(bbox) >= 4
        ):
            values = bbox[:4]

        else:
            return None

        try:
            x1, y1, x2, y2 = (
                float(value)
                for value in values
            )
        except (TypeError, ValueError):
            return None

        return (
            x1,
            y1,
            x2,
            y2,
        )

    @staticmethod
    def _horizontal_position(
        center_x: float,
        frame_width: int,
    ) -> str:
        if frame_width <= 0:
            return "unknown"

        ratio = (
            center_x
            / frame_width
        )

        if ratio < 0.33:
            return "left"

        if ratio > 0.67:
            return "right"

        return "center"

    @staticmethod
    def _vertical_position(
        center_y: float,
        frame_height: int,
    ) -> str:
        if frame_height <= 0:
            return "unknown"

        ratio = (
            center_y
            / frame_height
        )

        if ratio < 0.33:
            return "upper"

        if ratio > 0.67:
            return "lower"

        return "middle"

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    @staticmethod
    def _join_labels(
        labels: list[str],
    ) -> str:
        if not labels:
            return "no recognized objects"

        if len(labels) == 1:
            return f"a {labels[0]}"

        if len(labels) == 2:
            return (
                f"a {labels[0]} "
                f"and a {labels[1]}"
            )

        return (
            ", ".join(
                f"a {label}"
                for label in labels[:-1]
            )
            + f", and a {labels[-1]}"
        )