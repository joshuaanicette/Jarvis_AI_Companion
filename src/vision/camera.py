from __future__ import annotations

import glob
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.logger import logger


class Camera:
    """
    Camera wrapper for Raspberry Pi and Linux.

    Supported backends:
    - Picamera2 for Raspberry Pi CSI cameras
    - OpenCV/V4L2 for USB webcams

    read(), capture(), and get_frame() always return:
    - numpy.ndarray in BGR format
    - None when capture fails
    """

    def __init__(
        self,
        index: int | str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
        backend: str = "auto",
        warmup_seconds: float = 0.5,
        auto_detect: bool = True,
    ):
        self.index = index
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.backend = str(backend).strip().lower()
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self.auto_detect = bool(auto_detect)

        self.started = False
        self.active_backend: str | None = None
        self.active_device: int | str | None = None

        self._capture: cv2.VideoCapture | None = None
        self._picamera = None
        self._lock = threading.RLock()

    def start(self) -> bool:
        with self._lock:
            if self.is_open():
                return True

            self._release_resources()

            for backend_name in self._backend_order():
                if backend_name == "picamera2" and self._start_picamera2():
                    return True

                if backend_name == "opencv" and self._start_opencv():
                    return True

            logger.error(
                "No usable camera was found. "
                "Check the camera connection and backend configuration."
            )
            return False

    def stop(self) -> None:
        with self._lock:
            was_started = self.started
            self._release_resources()

            if was_started:
                logger.info("Camera stopped")

    def read(self) -> np.ndarray | None:
        with self._lock:
            if not self.started:
                logger.warning(
                    "Camera read requested before camera startup"
                )
                return None

            if self.active_backend == "picamera2":
                return self._read_picamera2()

            if self.active_backend == "opencv":
                return self._read_opencv()

            return None

    def capture(self) -> np.ndarray | None:
        return self.read()

    def get_frame(self) -> np.ndarray | None:
        return self.read()

    def is_open(self) -> bool:
        if not self.started:
            return False

        if self.active_backend == "picamera2":
            return self._picamera is not None

        if self.active_backend == "opencv":
            return bool(
                self._capture is not None
                and self._capture.isOpened()
            )

        return False

    def get_properties(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "open": self.is_open(),
            "backend": self.active_backend,
            "device": self.active_device,
            "requested_width": self.width,
            "requested_height": self.height,
            "requested_fps": self.fps,
        }

        if (
            self.active_backend == "opencv"
            and self._capture is not None
        ):
            properties.update(
                {
                    "actual_width": int(
                        self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                    ),
                    "actual_height": int(
                        self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    ),
                    "actual_fps": float(
                        self._capture.get(cv2.CAP_PROP_FPS)
                    ),
                }
            )

        return properties

    def _backend_order(self) -> tuple[str, ...]:
        if self.backend == "picamera2":
            return ("picamera2",)

        if self.backend == "opencv":
            return ("opencv",)

        # A Raspberry Pi CSI camera should use Picamera2.
        # Fall back to OpenCV for USB webcams.
        return (
            "picamera2",
            "opencv",
        )

    def _start_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2 # type: ignore
        except ImportError:
            logger.info(
                "Picamera2 is unavailable; trying OpenCV camera devices"
            )
            return False

        camera = None

        try:
            camera = Picamera2()

            configuration = camera.create_video_configuration(
                main={
                    "size": (
                        self.width,
                        self.height,
                    ),
                    "format": "RGB888",
                },
                controls={
                    "FrameRate": float(self.fps),
                },
            )

            camera.configure(configuration)
            camera.start()

            if self.warmup_seconds:
                time.sleep(self.warmup_seconds)

            frame = camera.capture_array("main")
            normalized = self._normalize_picamera_frame(frame)

            if normalized is None:
                raise RuntimeError(
                    "Picamera2 returned an invalid startup frame"
                )

            self._picamera = camera
            self._capture = None
            self.active_backend = "picamera2"
            self.active_device = "picamera2"
            self.started = True

            logger.info(
                "Picamera2 started with frame shape %s",
                normalized.shape,
            )
            return True

        except Exception as error:
            logger.warning(
                "Picamera2 startup failed: %s",
                error,
            )

            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass

                try:
                    camera.close()
                except Exception:
                    pass

            return False

    def _start_opencv(self) -> bool:
        candidates = self._opencv_candidates()

        if not candidates:
            logger.warning(
                "No /dev/video devices were found"
            )
            return False

        logger.info(
            "Testing OpenCV camera candidates: %s",
            candidates,
        )

        for candidate in candidates:
            capture = cv2.VideoCapture(
                candidate,
                cv2.CAP_V4L2,
            )

            if not capture.isOpened():
                capture.release()
                continue

            capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                float(self.width),
            )
            capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                float(self.height),
            )
            capture.set(
                cv2.CAP_PROP_FPS,
                float(self.fps),
            )
            capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

            frame = None

            for _ in range(10):
                success, candidate_frame = capture.read()

                if success:
                    frame = self._normalize_opencv_frame(
                        candidate_frame
                    )

                if frame is not None:
                    break

                time.sleep(0.05)

            if frame is None:
                capture.release()
                continue

            self._capture = capture
            self._picamera = None
            self.active_backend = "opencv"
            self.active_device = candidate
            self.started = True

            logger.info(
                "OpenCV camera started on %s with frame shape %s",
                candidate,
                frame.shape,
            )
            return True

        logger.warning(
            "No OpenCV/V4L2 device returned a valid frame"
        )
        return False

    def _read_picamera2(self) -> np.ndarray | None:
        if self._picamera is None:
            return None

        try:
            frame = self._picamera.capture_array("main")
        except Exception as error:
            logger.warning(
                "Picamera2 frame capture failed: %s",
                error,
            )
            return None

        return self._normalize_picamera_frame(frame)

    def _read_opencv(self) -> np.ndarray | None:
        if self._capture is None:
            return None

        try:
            success, frame = self._capture.read()
        except Exception as error:
            logger.warning(
                "OpenCV camera read failed: %s",
                error,
            )
            return None

        if not success:
            return None

        return self._normalize_opencv_frame(frame)

    def _opencv_candidates(self) -> list[int | str]:
        candidates: list[int | str] = []

        if self.index is not None:
            candidates.append(self.index)

            if isinstance(self.index, int):
                path = f"/dev/video{self.index}"

                if Path(path).exists():
                    candidates.append(path)

        if self.auto_detect:
            candidates.extend(
                sorted(
                    glob.glob("/dev/video*"),
                    key=self._video_sort_key,
                )
            )

        unique: list[int | str] = []
        seen: set[str] = set()

        for candidate in candidates:
            marker = str(candidate)

            if marker in seen:
                continue

            seen.add(marker)
            unique.append(candidate)

        return unique

    def _release_resources(self) -> None:
        capture = self._capture
        picamera = self._picamera

        self._capture = None
        self._picamera = None
        self.started = False
        self.active_backend = None
        self.active_device = None

        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

        if picamera is not None:
            try:
                picamera.stop()
            except Exception:
                pass

            try:
                picamera.close()
            except Exception:
                pass

    @staticmethod
    def _normalize_picamera_frame(
        frame,
    ) -> np.ndarray | None:
        if not isinstance(frame, np.ndarray):
            return None

        if frame.size == 0:
            return None

        if frame.ndim == 2:
            return cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

        if frame.ndim != 3:
            return None

        channels = frame.shape[2]

        if channels == 3:
            # Picamera2 configuration uses RGB888.
            converted = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )
            return np.ascontiguousarray(converted)

        if channels == 4:
            converted = cv2.cvtColor(
                frame,
                cv2.COLOR_RGBA2BGR,
            )
            return np.ascontiguousarray(converted)

        return None

    @staticmethod
    def _normalize_opencv_frame(
        frame,
    ) -> np.ndarray | None:
        if not isinstance(frame, np.ndarray):
            return None

        if frame.size == 0:
            return None

        if frame.ndim == 2:
            converted = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )
            return np.ascontiguousarray(converted)

        if frame.ndim != 3:
            return None

        channels = frame.shape[2]

        if channels == 3:
            return np.ascontiguousarray(frame)

        if channels == 4:
            converted = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR,
            )
            return np.ascontiguousarray(converted)

        if channels == 1:
            converted = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )
            return np.ascontiguousarray(converted)

        return None

    @staticmethod
    def _video_sort_key(
        path: str,
    ) -> tuple[int, str]:
        name = Path(path).name
        suffix = name.removeprefix("video")

        try:
            number = int(suffix)
        except ValueError:
            number = 9999

        return (
            number,
            path,
        )

    def __enter__(self):
        if not self.start():
            raise RuntimeError(
                "The camera could not be started."
            )

        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        self.stop()
