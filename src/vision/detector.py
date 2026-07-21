from __future__ import annotations

from typing import Any

import numpy as np

from src.core.logger import logger


class ObjectDetector:
    """
    Ultralytics YOLO object detector.

    The detector accepts a NumPy image and returns a list of
    dictionaries compatible with SceneAnalyzer, RoomScanner,
    and VisionManager.

    Each detection has this format:

    {
        "label": "person",
        "confidence": 0.91,
        "bbox": [x1, y1, x2, y2],
        "width": 120.0,
        "height": 300.0,
        "area": 36000.0,
    }
    """

    def __init__(
        self,
        model_name: str | None = None,
        model_path: str | None = None,
        confidence: float = 0.45,
        image_size: int = 416,
        device: str = "cpu",
        **kwargs,
    ):
        selected_model = (
            model_path
            or model_name
            or "yolo11n.pt"
        )

        self.model_name = str(
            selected_model
        )

        self.model_path = self.model_name
        self.confidence = float(
            confidence
        )
        self.image_size = int(
            image_size
        )
        self.device = str(
            device
        )

        self.extra_options = dict(
            kwargs
        )

        self.latest_results = None
        self.latest_detections: list[
            dict[str, Any]
        ] = []

        self.model = self._load_model(
            self.model_path
        )

    def _load_model(
        self,
        model_path: str,
    ):
        try:
            from ultralytics import YOLO

        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. "
                "Install it with: pip install ultralytics"
            ) from error

        try:
            model = YOLO(
                model_path
            )

        except Exception as error:
            raise RuntimeError(
                f"Could not load YOLO model: {model_path}"
            ) from error

        logger.info(
            "YOLO detector loaded: %s",
            model_path,
        )

        return model

    def detect(
        self,
        frame,
    ) -> list[dict[str, Any]]:
        """
        Detect objects in one image.
        """

        image = self._extract_image(
            frame
        )

        if image is None:
            logger.warning(
                "Object detector received an invalid image"
            )
            return []

        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )

        except TypeError:
            results = self.model.predict(
                image,
                conf=self.confidence,
                verbose=False,
            )

        except Exception as error:
            logger.exception(
                "YOLO object detection failed: %s",
                error,
            )
            return []

        self.latest_results = results

        detections = self._convert_results(
            results
        )

        self.latest_detections = detections

        return detections

    def predict(
        self,
        frame,
    ) -> list[dict[str, Any]]:
        return self.detect(
            frame
        )

    def process(
        self,
        frame,
    ) -> list[dict[str, Any]]:
        return self.detect(
            frame
        )

    def draw_detections(
        self,
        frame,
        detections: list[Any] | None = None,
    ) -> np.ndarray:
        """
        Draw boxes and labels on a camera image.
        """

        image = self._extract_image(
            frame
        )

        if image is None:
            raise ValueError(
                "Cannot annotate an invalid image."
            )

        annotated = image.copy()

        if detections is None:
            detections = self.latest_detections

        try:
            import cv2

        except ImportError:
            return annotated

        for detection in detections:
            bbox = self._get_bbox(
                detection
            )

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            label = self._get_label(
                detection
            )

            confidence = self._get_confidence(
                detection
            )

            text = label

            if confidence is not None:
                text = (
                    f"{label} {confidence:.0%}"
                )

            cv2.rectangle(
                annotated,
                (
                    int(x1),
                    int(y1),
                ),
                (
                    int(x2),
                    int(y2),
                ),
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

            cv2.putText(
                annotated,
                text,
                (
                    int(x1),
                    max(
                        20,
                        int(y1) - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

        return annotated

    def annotate(
        self,
        frame,
        detections: list[Any] | None = None,
    ) -> np.ndarray:
        return self.draw_detections(
            frame,
            detections,
        )

    def render(
        self,
        frame,
        detections: list[Any] | None = None,
    ) -> np.ndarray:
        return self.draw_detections(
            frame,
            detections,
        )

    @classmethod
    def _extract_image(
        cls,
        value,
    ) -> np.ndarray | None:
        if value is None:
            return None

        if isinstance(
            value,
            np.ndarray,
        ):
            return cls._normalize_image(
                value
            )

        if isinstance(
            value,
            tuple,
        ):
            if (
                len(value) >= 2
                and bool(value[0])
            ):
                return cls._extract_image(
                    value[1]
                )

            return None

        if isinstance(
            value,
            dict,
        ):
            for key in (
                "image",
                "frame",
                "array",
                "orig_img",
            ):
                if key not in value:
                    continue

                image = cls._extract_image(
                    value[key]
                )

                if image is not None:
                    return image

        for attribute in (
            "image",
            "frame",
            "array",
            "orig_img",
        ):
            candidate = getattr(
                value,
                attribute,
                None,
            )

            if candidate is value:
                continue

            image = cls._extract_image(
                candidate
            )

            if image is not None:
                return image

        return None

    @staticmethod
    def _normalize_image(
        image: np.ndarray,
    ) -> np.ndarray | None:
        if image.size == 0:
            return None

        if image.ndim == 2:
            try:
                import cv2

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR,
                )

            except Exception:
                return None

        if image.ndim != 3:
            return None

        channels = image.shape[2]

        if channels == 4:
            try:
                import cv2

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2BGR,
                )

            except Exception:
                return None

        elif channels == 1:
            try:
                import cv2

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2BGR,
                )

            except Exception:
                return None

        elif channels != 3:
            return None

        return np.ascontiguousarray(
            image
        )

    def _convert_results(
        self,
        results,
    ) -> list[dict[str, Any]]:
        detections: list[
            dict[str, Any]
        ] = []

        if results is None:
            return detections

        try:
            result_items = list(
                results
            )

        except TypeError:
            result_items = [
                results
            ]

        for result in result_items:
            names = getattr(
                result,
                "names",
                {},
            )

            boxes = getattr(
                result,
                "boxes",
                None,
            )

            if boxes is None:
                continue

            try:
                box_items = list(
                    boxes
                )

            except TypeError:
                continue

            for box in box_items:
                detection = self._convert_box(
                    box,
                    names,
                )

                if detection is not None:
                    detections.append(
                        detection
                    )

        return detections

    @staticmethod
    def _convert_box(
        box,
        names,
    ) -> dict[str, Any] | None:
        try:
            class_id = ObjectDetector._extract_scalar_int(
                box.cls
            )

            confidence = ObjectDetector._extract_scalar_float(
                box.conf
            )

            coordinates = box.xyxy

            if hasattr(
                coordinates,
                "detach",
            ):
                coordinates = coordinates.detach()

            if hasattr(
                coordinates,
                "cpu",
            ):
                coordinates = coordinates.cpu()

            if hasattr(
                coordinates,
                "numpy",
            ):
                coordinates = coordinates.numpy()

            coordinates = np.asarray(
                coordinates
            ).reshape(
                -1
            )

            if coordinates.size < 4:
                return None

            x1 = float(
                coordinates[0]
            )
            y1 = float(
                coordinates[1]
            )
            x2 = float(
                coordinates[2]
            )
            y2 = float(
                coordinates[3]
            )

            if isinstance(
                names,
                dict,
            ):
                label = str(
                    names.get(
                        class_id,
                        class_id,
                    )
                )

            else:
                try:
                    label = str(
                        names[class_id]
                    )

                except Exception:
                    label = str(
                        class_id
                    )

            width = max(
                0.0,
                x2 - x1,
            )

            height = max(
                0.0,
                y2 - y1,
            )

            area = (
                width
                * height
            )

            return {
                "label": label,
                "name": label,
                "class_name": label,
                "class_id": class_id,
                "confidence": confidence,
                "score": confidence,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "area": area,
            }

        except Exception as error:
            logger.warning(
                "Could not convert YOLO detection: %s",
                error,
            )
            return None

    @staticmethod
    def _extract_scalar_int(
        value,
    ) -> int:
        if hasattr(
            value,
            "detach",
        ):
            value = value.detach()

        if hasattr(
            value,
            "cpu",
        ):
            value = value.cpu()

        if hasattr(
            value,
            "numpy",
        ):
            value = value.numpy()

        array = np.asarray(
            value
        ).reshape(
            -1
        )

        if array.size == 0:
            raise ValueError(
                "The class ID value is empty."
            )

        return int(
            array[0]
        )

    @staticmethod
    def _extract_scalar_float(
        value,
    ) -> float:
        if hasattr(
            value,
            "detach",
        ):
            value = value.detach()

        if hasattr(
            value,
            "cpu",
        ):
            value = value.cpu()

        if hasattr(
            value,
            "numpy",
        ):
            value = value.numpy()

        array = np.asarray(
            value
        ).reshape(
            -1
        )

        if array.size == 0:
            raise ValueError(
                "The confidence value is empty."
            )

        return float(
            array[0]
        )

    @staticmethod
    def _get_label(
        detection: Any,
    ) -> str:
        if isinstance(
            detection,
            dict,
        ):
            for key in (
                "label",
                "name",
                "class_name",
                "class",
            ):
                value = detection.get(
                    key
                )

                if value is not None:
                    return str(
                        value
                    )

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
                return str(
                    value
                )

        return "object"

    @staticmethod
    def _get_confidence(
        detection: Any,
    ) -> float | None:
        value = None

        if isinstance(
            detection,
            dict,
        ):
            for key in (
                "confidence",
                "score",
                "conf",
            ):
                if key in detection:
                    value = detection[
                        key
                    ]
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
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
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
        if isinstance(
            detection,
            dict,
        ):
            box = (
                detection.get(
                    "bbox"
                )
                or detection.get(
                    "box"
                )
                or detection.get(
                    "bounding_box"
                )
            )

        else:
            box = (
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

        if isinstance(
            box,
            dict,
        ):
            values = (
                box.get(
                    "x1",
                    box.get(
                        "left"
                    ),
                ),
                box.get(
                    "y1",
                    box.get(
                        "top"
                    ),
                ),
                box.get(
                    "x2",
                    box.get(
                        "right"
                    ),
                ),
                box.get(
                    "y2",
                    box.get(
                        "bottom"
                    ),
                ),
            )

        elif isinstance(
            box,
            (
                list,
                tuple,
            ),
        ) and len(
            box
        ) >= 4:
            values = box[:4]

        else:
            return None

        try:
            x1, y1, x2, y2 = (
                float(value)
                for value in values
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        return (
            x1,
            y1,
            x2,
            y2,
        )


Detector = ObjectDetector