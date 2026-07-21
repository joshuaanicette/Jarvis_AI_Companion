from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from src.core.logger import logger
from src.vision.motion_detector import MotionDetector


class VisionManager:
    """
    Coordinate the camera, detector, scene analyzer, room scanner,
    live display, and background vision processing.
    """

    def __init__(
        self,
        camera,
        detector,
        analyzer,
        room_scanner=None,
    ):
        self.camera = camera
        self.detector = detector
        self.analyzer = analyzer
        self.room_scanner = room_scanner

        self.running = False
        self.started = False

        self.latest_frame: np.ndarray | None = None
        self.latest_detections: list[Any] = []
        self.latest_scene = ""
        self.latest_scan = None

        self.pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        self._show_camera = False
        self._detection_interval = 0.3
        self._motion_triggered = True
        self.motion_detector = MotionDetector()

    def start(self) -> bool:
        with self._lock:
            if self.started:
                return True

            try:
                started = self.camera.start()
            except Exception as error:
                logger.exception(
                    "Could not start the camera: %s",
                    error,
                )
                return False

            if started is False:
                logger.error(
                    "Camera start method returned False"
                )
                return False

            self.started = True
            logger.info("Vision camera started")
            return True

    def stop(self) -> None:
        self.stop_continuous()

        with self._lock:
            if not self.started:
                return

            try:
                self.camera.stop()
            except Exception as error:
                logger.warning(
                    "Could not stop camera cleanly: %s",
                    error,
                )

            self.started = False
            self.latest_frame = None
            logger.info("Vision camera stopped")

    def start_continuous(
        self,
        show_camera: bool = True,
        detection_interval: float = 0.3,
        motion_triggered: bool = True,
    ) -> bool:
        with self._lock:
            if self.running:
                logger.info(
                    "Continuous vision is already running"
                )
                return False

            if not self.started and not self.start():
                return False

            self._show_camera = bool(show_camera)
            self._detection_interval = max(
                0.1,
                float(detection_interval),
            )
            self._motion_triggered = bool(motion_triggered)
            self.motion_detector.reset()

            self._stop_event.clear()
            self.pause_event.clear()
            self.running = True

            self._thread = threading.Thread(
                target=self._continuous_loop,
                daemon=True,
                name="JoeVisionProcessor",
            )
            self._thread.start()

            logger.info(
                "Continuous vision started with interval %.2f seconds",
                self._detection_interval,
            )
            return True

    def stop_continuous(self) -> None:
        thread = self._thread

        if not self.running and thread is None:
            return

        logger.info("Stopping continuous vision")

        self._stop_event.set()
        self.pause_event.clear()
        self.running = False

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3.0)

        self._thread = None
        self._close_display_window()

        logger.info("Continuous vision stopped")

    def pause(self) -> None:
        if not self.pause_event.is_set():
            self.pause_event.set()
            logger.info("Vision processing paused")

    def resume(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            logger.info("Vision processing resumed")

    def process_once(
        self,
        show_camera: bool = False,
        display_seconds: float = 3.0,
    ) -> str:
        if not self.started and not self.start():
            raise RuntimeError(
                "The camera could not be started."
            )

        frame = self._read_frame()

        if frame is None:
            raise RuntimeError(
                "The camera did not return a valid frame."
            )

        detections = self._detect_objects(frame)
        scene = self._analyze_scene(detections)
        scan = self._scan_room(
            frame=frame,
            detections=detections,
        )

        self._store_results(
            frame=frame,
            detections=detections,
            scene=scene,
            scan=scan,
        )

        logger.info(
            "Vision scene: %s",
            scene,
        )

        if show_camera:
            annotated = self._annotate_frame(
                frame,
                detections,
            )
            self._display_frame(
                annotated,
                display_seconds,
            )

        return scene

    def get_room_summary(self) -> str:
        with self._lock:
            scan = self.latest_scan
            scene = self.latest_scene

        if scan is None:
            return (
                scene
                or (
                    "I do not have a recent room scan. "
                    "Ask me to scan the room first."
                )
            )

        if isinstance(scan, str):
            return scan

        if isinstance(scan, dict):
            for key in (
                "summary",
                "description",
                "scene",
            ):
                value = scan.get(key)

                if value:
                    return str(value)

        return (
            scene
            or "I completed the scan, but no summary was available."
        )

    def describe_object(
        self,
        object_label: str,
    ) -> str:
        requested = object_label.strip().lower()

        if not requested:
            return (
                "Please specify which object you want me to measure."
            )

        with self._lock:
            scan = self.latest_scan
            detections = list(
                self.latest_detections
            )

        if scan is None and not detections:
            return (
                "I do not have a recent room scan. "
                "Ask me to scan the room first."
            )

        if self.room_scanner is not None:
            for name in (
                "describe_object",
                "get_object_description",
                "measure_object",
            ):
                method = getattr(
                    self.room_scanner,
                    name,
                    None,
                )

                if not callable(method):
                    continue

                for args in (
                    (scan, requested),
                    (requested, scan),
                    (requested,),
                ):
                    try:
                        result = method(*args)
                    except TypeError:
                        continue

                    if result:
                        return str(result)

        detection = self._find_detection(
            requested,
            detections,
        )

        if detection is None:
            return (
                f"I did not find a {object_label} "
                "in the latest camera scan."
            )

        return self._format_detection_description(
            detection
        )

    def get_largest_object(self) -> str:
        with self._lock:
            detections = list(
                self.latest_detections
            )

        if not detections:
            return (
                "I do not have any detected objects to compare."
            )

        largest = max(
            detections,
            key=self._detection_area,
        )

        label = self._detection_label(largest)
        area = self._detection_area(largest)

        return (
            f"The largest detected object is {label}. "
            f"Its detected image area is approximately "
            f"{area:.0f} square pixels."
        )

    def get_latest_scene(self) -> str:
        with self._lock:
            return self.latest_scene

    def get_latest_detections(
        self,
    ) -> list[Any]:
        with self._lock:
            return list(
                self.latest_detections
            )

    def _continuous_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.pause_event.is_set():
                    self._stop_event.wait(0.1)
                    continue

                started_at = time.monotonic()

                try:
                    frame = self._read_frame()

                    if frame is None:
                        logger.warning(
                            "Camera returned no valid frame"
                        )
                        self._stop_event.wait(0.2)
                        continue

                    run_detection = (
                        not self._motion_triggered
                        or self.motion_detector.should_run_detection(
                            frame
                        )
                    )

                    if run_detection:
                        detections = self._detect_objects(
                            frame
                        )
                        scene = self._analyze_scene(
                            detections
                        )
                        scan = self._scan_room(
                            frame=frame,
                            detections=detections,
                        )

                        self._store_results(
                            frame=frame,
                            detections=detections,
                            scene=scene,
                            scan=scan,
                        )

                        logger.info(
                            "Vision scene: %s",
                            scene,
                        )
                    else:
                        with self._lock:
                            detections = list(
                                self.latest_detections
                            )

                    if self._show_camera:
                        annotated = self._annotate_frame(
                            frame,
                            detections,
                        )

                        if self._show_live_frame(
                            annotated
                        ):
                            self._stop_event.set()
                            break

                except Exception as error:
                    logger.exception(
                        "Continuous vision frame failed: %s",
                        error,
                    )
                    self._stop_event.wait(0.25)

                elapsed = (
                    time.monotonic()
                    - started_at
                )
                remaining = (
                    self._detection_interval
                    - elapsed
                )

                if remaining > 0:
                    self._stop_event.wait(
                        remaining
                    )

        finally:
            self.running = False
            self._close_display_window()

    def _store_results(
        self,
        frame: np.ndarray,
        detections: list[Any],
        scene: str,
        scan,
    ) -> None:
        with self._lock:
            self.latest_frame = frame
            self.latest_detections = detections
            self.latest_scene = scene
            self.latest_scan = scan

    def _read_frame(
        self,
    ) -> np.ndarray | None:
        for method_name in (
            "read",
            "capture",
            "get_frame",
        ):
            method = getattr(
                self.camera,
                method_name,
                None,
            )

            if not callable(method):
                continue

            result = method()
            return self._extract_frame(result)

        raise AttributeError(
            "The camera does not provide read(), "
            "capture(), or get_frame()."
        )

    @classmethod
    def _extract_frame(
        cls,
        value,
    ) -> np.ndarray | None:
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            return cls._normalize_frame(value)

        if isinstance(value, tuple):
            if len(value) >= 2 and bool(value[0]):
                return cls._extract_frame(
                    value[1]
                )

            return None

        if isinstance(value, dict):
            for key in (
                "frame",
                "image",
                "array",
                "orig_img",
            ):
                if key in value:
                    frame = cls._extract_frame(
                        value[key]
                    )

                    if frame is not None:
                        return frame

        for attribute in (
            "frame",
            "image",
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

            frame = cls._extract_frame(
                candidate
            )

            if frame is not None:
                return frame

        return None

    @staticmethod
    def _normalize_frame(
        frame: np.ndarray,
    ) -> np.ndarray | None:
        if frame.size == 0:
            return None

        if frame.ndim == 2:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

        if frame.ndim != 3:
            return None

        channels = frame.shape[2]

        if channels == 4:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR,
            )
        elif channels == 1:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )
        elif channels != 3:
            return None

        return np.ascontiguousarray(
            frame
        )

    def _detect_objects(
        self,
        frame: np.ndarray,
    ) -> list[Any]:
        for method_name in (
            "detect",
            "predict",
            "process",
        ):
            method = getattr(
                self.detector,
                method_name,
                None,
            )

            if not callable(method):
                continue

            result = method(frame)
            return self._extract_detections(
                result
            )

        raise AttributeError(
            "The detector does not provide detect(), "
            "predict(), or process()."
        )

    @staticmethod
    def _extract_detections(
        result,
    ) -> list[Any]:
        if result is None:
            return []

        if isinstance(result, list):
            return result

        if isinstance(result, tuple):
            return list(result)

        if isinstance(result, dict):
            for key in (
                "detections",
                "objects",
                "results",
            ):
                items = result.get(key)

                if isinstance(items, list):
                    return items

            return [result]

        for attribute in (
            "detections",
            "objects",
            "results",
        ):
            items = getattr(
                result,
                attribute,
                None,
            )

            if items is not None:
                try:
                    return list(items)
                except TypeError:
                    pass

        try:
            return list(result)
        except TypeError:
            return [result]

    def _analyze_scene(
        self,
        detections: list[Any],
    ) -> str:
        for method_name in (
            "analyze",
            "describe",
            "summarize",
        ):
            method = getattr(
                self.analyzer,
                method_name,
                None,
            )

            if not callable(method):
                continue

            result = method(detections)

            if result:
                return str(result)

        if not detections:
            return (
                "I opened the camera, but I did not "
                "recognize any objects."
            )

        labels = [
            self._detection_label(item)
            for item in detections
        ]

        return (
            "I can see "
            + self._join_labels(labels)
            + "."
        )

    def _scan_room(
        self,
        frame: np.ndarray,
        detections: list[Any],
    ):
        if self.room_scanner is None:
            return {
                "summary": self._analyze_scene(
                    detections
                ),
                "objects": detections,
            }

        for method_name in (
            "scan",
            "analyze",
            "process",
        ):
            method = getattr(
                self.room_scanner,
                method_name,
                None,
            )

            if not callable(method):
                continue

            attempts = (
                lambda: method(
                    frame=frame,
                    detections=detections,
                ),
                lambda: method(
                    frame,
                    detections,
                ),
                lambda: method(detections),
                lambda: method(frame),
            )

            for attempt in attempts:
                try:
                    result = attempt()
                except TypeError:
                    continue

                if result is not None:
                    return result

        return {
            "summary": self._analyze_scene(
                detections
            ),
            "objects": detections,
        }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[Any],
    ) -> np.ndarray:
        original = frame.copy()

        for method_name in (
            "draw_detections",
            "annotate",
            "render",
        ):
            method = getattr(
                self.detector,
                method_name,
                None,
            )

            if not callable(method):
                continue

            for args in (
                (
                    original.copy(),
                    detections,
                ),
                (
                    original.copy(),
                ),
            ):
                try:
                    result = method(*args)
                except TypeError:
                    continue
                except Exception as error:
                    logger.warning(
                        "Detector annotation failed: %s",
                        error,
                    )
                    continue

                converted = self._extract_annotated_frame(
                    result
                )

                if converted is not None:
                    return converted

        return self._draw_detections(
            original,
            detections,
        )

    @classmethod
    def _extract_annotated_frame(
        cls,
        result,
    ) -> np.ndarray | None:
        frame = cls._extract_frame(result)

        if frame is not None:
            return frame

        if isinstance(result, (list, tuple)):
            for item in result:
                frame = cls._extract_annotated_frame(
                    item
                )

                if frame is not None:
                    return frame

        plot_method = getattr(
            result,
            "plot",
            None,
        )

        if callable(plot_method):
            try:
                return cls._extract_frame(
                    plot_method()
                )
            except Exception:
                return None

        return None

    @classmethod
    def _draw_detections(
        cls,
        frame: np.ndarray,
        detections: list[Any],
    ) -> np.ndarray:
        for detection in detections:
            box = cls._extract_box(
                detection
            )

            if box is None:
                continue

            x1, y1, x2, y2 = box
            label = cls._detection_label(
                detection
            )
            confidence = cls._detection_confidence(
                detection
            )

            text = label

            if confidence is not None:
                text = (
                    f"{label} {confidence:.0%}"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                text,
                (
                    x1,
                    max(20, y1 - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        return frame

    @staticmethod
    def _display_frame(
        frame: np.ndarray,
        display_seconds: float,
    ) -> None:
        if not isinstance(
            frame,
            np.ndarray,
        ):
            return

        try:
            end_time = (
                time.monotonic()
                + max(
                    0.1,
                    float(display_seconds),
                )
            )

            while time.monotonic() < end_time:
                cv2.imshow(
                    "Joe Vision",
                    frame,
                )

                key = cv2.waitKey(30) & 0xFF

                if key in {
                    ord("q"),
                    27,
                }:
                    break

        except Exception as error:
            logger.warning(
                "Could not display camera frame: %s",
                error,
            )
        finally:
            VisionManager._close_display_window()

    @staticmethod
    def _show_live_frame(
        frame: np.ndarray,
    ) -> bool:
        if not isinstance(
            frame,
            np.ndarray,
        ):
            return False

        try:
            cv2.imshow(
                "Joe Vision",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            return key in {
                ord("q"),
                27,
            }

        except Exception as error:
            logger.warning(
                "Could not show live camera frame: %s",
                error,
            )
            return False

    @staticmethod
    def _close_display_window() -> None:
        try:
            cv2.destroyWindow(
                "Joe Vision"
            )
        except Exception:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    @classmethod
    def _find_detection(
        cls,
        requested: str,
        detections: list[Any],
    ):
        for detection in detections:
            label = cls._detection_label(
                detection
            ).lower()

            if (
                requested == label
                or requested in label
                or label in requested
            ):
                return detection

        return None

    @classmethod
    def _format_detection_description(
        cls,
        detection: Any,
    ) -> str:
        label = cls._detection_label(
            detection
        )
        confidence = cls._detection_confidence(
            detection
        )
        width, height = cls._detection_dimensions(
            detection
        )
        area = cls._detection_area(
            detection
        )

        parts = [
            f"I detected a {label}",
        ]

        if confidence is not None:
            parts.append(
                f"with {confidence:.0%} confidence"
            )

        if (
            width is not None
            and height is not None
        ):
            parts.append(
                f"with a detected width of {width:.0f} pixels "
                f"and height of {height:.0f} pixels"
            )

        if area > 0:
            parts.append(
                f"covering approximately {area:.0f} square pixels"
            )

        return (
            ", ".join(parts)
            + "."
        )

    @staticmethod
    def _detection_label(
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
    def _detection_confidence(
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

    @classmethod
    def _detection_area(
        cls,
        detection: Any,
    ) -> float:
        width, height = cls._detection_dimensions(
            detection
        )

        if (
            width is None
            or height is None
        ):
            return 0.0

        return (
            max(0.0, width)
            * max(0.0, height)
        )

    @classmethod
    def _detection_dimensions(
        cls,
        detection: Any,
    ) -> tuple[
        float | None,
        float | None,
    ]:
        box = cls._extract_box(
            detection
        )

        if box is not None:
            x1, y1, x2, y2 = box

            return (
                float(x2 - x1),
                float(y2 - y1),
            )

        if isinstance(detection, dict):
            width = detection.get("width")
            height = detection.get("height")
        else:
            width = getattr(
                detection,
                "width",
                None,
            )
            height = getattr(
                detection,
                "height",
                None,
            )

        try:
            if (
                width is not None
                and height is not None
            ):
                return (
                    float(width),
                    float(height),
                )
        except (TypeError, ValueError):
            pass

        return (
            None,
            None,
        )

    @staticmethod
    def _extract_box(
        detection: Any,
    ) -> tuple[int, int, int, int] | None:
        box = None

        if isinstance(detection, dict):
            box = (
                detection.get("bbox")
                or detection.get("box")
                or detection.get("bounding_box")
            )

            if box is None and all(
                key in detection
                for key in (
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                )
            ):
                box = (
                    detection["x1"],
                    detection["y1"],
                    detection["x2"],
                    detection["y2"],
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

        if isinstance(box, dict):
            values = (
                box.get(
                    "x1",
                    box.get("left"),
                ),
                box.get(
                    "y1",
                    box.get("top"),
                ),
                box.get(
                    "x2",
                    box.get("right"),
                ),
                box.get(
                    "y2",
                    box.get("bottom"),
                ),
            )
        elif (
            isinstance(
                box,
                (list, tuple),
            )
            and len(box) >= 4
        ):
            values = box[:4]
        else:
            return None

        try:
            x1, y1, x2, y2 = (
                int(float(value))
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
    def _join_labels(
        labels: list[str],
    ) -> str:
        cleaned = [
            label
            for label in labels
            if label
        ]

        if not cleaned:
            return "no recognized objects"

        if len(cleaned) == 1:
            return f"a {cleaned[0]}"

        if len(cleaned) == 2:
            return (
                f"a {cleaned[0]} and a {cleaned[1]}"
            )

        return (
            ", ".join(
                f"a {label}"
                for label in cleaned[:-1]
            )
            + f", and a {cleaned[-1]}"
        )