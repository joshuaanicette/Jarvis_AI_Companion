from dataclasses import dataclass
from typing import Any


@dataclass
class ObjectMeasurement:
    label: str
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int

    pixel_width: int
    pixel_height: int
    pixel_area: int
    frame_area_percentage: float

    estimated_width_cm: float | None
    estimated_height_cm: float | None
    estimated_front_area_cm2: float | None
    estimated_distance_cm: float | None

    horizontal_position: str
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "box": [
                self.x1,
                self.y1,
                self.x2,
                self.y2,
            ],
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "pixel_area": self.pixel_area,
            "frame_area_percentage": self.frame_area_percentage,
            "estimated_width_cm": self.estimated_width_cm,
            "estimated_height_cm": self.estimated_height_cm,
            "estimated_front_area_cm2": (
                self.estimated_front_area_cm2
            ),
            "estimated_distance_cm": self.estimated_distance_cm,
            "horizontal_position": self.horizontal_position,
            "track_id": self.track_id,
        }


class ObjectMeasurementEstimator:
    """
    Estimates object dimensions and distance from a monocular camera.

    Pixel measurements are taken directly from the detector's
    bounding box.

    Physical measurements are approximate and depend on:
    - camera calibration
    - object orientation
    - bounding-box accuracy
    - assumed reference width
    """

    KNOWN_WIDTHS_CM = {
        "person": 45.0,
        "chair": 48.0,
        "bottle": 7.0,
        "cup": 8.5,
        "cell phone": 7.5,
        "laptop": 33.0,
        "keyboard": 44.0,
        "mouse": 6.5,
        "book": 15.0,
        "tv": 100.0,
        "monitor": 55.0,
        "backpack": 32.0,
        "remote": 5.0,
        "clock": 25.0,
    }

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        focal_length_x_pixels: float = 700.0,
        focal_length_y_pixels: float | None = None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.focal_length_x_pixels = (
            focal_length_x_pixels
        )

        self.focal_length_y_pixels = (
            focal_length_y_pixels
            if focal_length_y_pixels is not None
            else focal_length_x_pixels
        )

    def measure_detection(
        self,
        detection: dict[str, Any],
    ) -> ObjectMeasurement:
        label = str(
            detection.get(
                "label",
                detection.get(
                    "class_name",
                    "object",
                ),
            )
        )

        confidence = float(
            detection.get(
                "confidence",
                detection.get(
                    "score",
                    0.0,
                ),
            )
        )

        box = detection.get(
            "bbox",
            detection.get("box"),
        )

        if box is None or len(box) != 4:
            raise ValueError(
                "Detection must contain a four-value "
                "bbox or box."
            )

        x1, y1, x2, y2 = [
            int(value)
            for value in box
        ]

        pixel_width = max(
            1,
            x2 - x1,
        )

        pixel_height = max(
            1,
            y2 - y1,
        )

        pixel_area = (
            pixel_width
            * pixel_height
        )

        frame_area = max(
            1,
            self.frame_width
            * self.frame_height,
        )

        frame_area_percentage = round(
            pixel_area
            / frame_area
            * 100,
            2,
        )

        center_x = (
            x1
            + pixel_width / 2
        )

        horizontal_position = (
            self._horizontal_position(
                center_x
            )
        )

        known_width = (
            self.KNOWN_WIDTHS_CM.get(
                label.lower()
            )
        )

        estimated_distance = None
        estimated_width = None
        estimated_height = None
        estimated_front_area = None

        if known_width is not None:
            estimated_distance = (
                self.estimate_distance_cm(
                    known_width_cm=known_width,
                    pixel_width=pixel_width,
                )
            )

            estimated_width = round(
                known_width,
                1,
            )

            estimated_height = (
                self.estimate_height_cm(
                    pixel_height=pixel_height,
                    distance_cm=estimated_distance,
                )
            )

            estimated_front_area = round(
                estimated_width
                * estimated_height,
                1,
            )

        track_id = detection.get(
            "track_id"
        )

        if track_id is not None:
            track_id = int(
                track_id
            )

        return ObjectMeasurement(
            label=label,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            pixel_area=pixel_area,
            frame_area_percentage=frame_area_percentage,
            estimated_width_cm=estimated_width,
            estimated_height_cm=estimated_height,
            estimated_front_area_cm2=estimated_front_area,
            estimated_distance_cm=estimated_distance,
            horizontal_position=horizontal_position,
            track_id=track_id,
        )

    def estimate_distance_cm(
        self,
        known_width_cm: float,
        pixel_width: int,
    ) -> float:
        if pixel_width <= 0:
            raise ValueError(
                "Pixel width must be greater than zero."
            )

        distance = (
            known_width_cm
            * self.focal_length_x_pixels
            / pixel_width
        )

        return round(
            distance,
            1,
        )

    def estimate_width_cm(
        self,
        pixel_width: int,
        distance_cm: float,
    ) -> float:
        if self.focal_length_x_pixels <= 0:
            raise ValueError(
                "Horizontal focal length must "
                "be greater than zero."
            )

        width = (
            pixel_width
            * distance_cm
            / self.focal_length_x_pixels
        )

        return round(
            width,
            1,
        )

    def estimate_height_cm(
        self,
        pixel_height: int,
        distance_cm: float,
    ) -> float:
        if self.focal_length_y_pixels <= 0:
            raise ValueError(
                "Vertical focal length must "
                "be greater than zero."
            )

        height = (
            pixel_height
            * distance_cm
            / self.focal_length_y_pixels
        )

        return round(
            height,
            1,
        )

    def _horizontal_position(
        self,
        center_x: float,
    ) -> str:
        left_boundary = (
            self.frame_width / 3
        )

        right_boundary = (
            self.frame_width
            * 2 / 3
        )

        if center_x < left_boundary:
            return "left"

        if center_x > right_boundary:
            return "right"

        return "center"