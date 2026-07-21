from __future__ import annotations

import time

import cv2
import numpy as np


class MotionDetector:
    def __init__(
        self,
        threshold: int = 22,
        minimum_area: int = 2500,
        blur_size: int = 21,
        refresh_seconds: float = 5.0,
    ):
        self.threshold = int(threshold)
        self.minimum_area = int(minimum_area)
        self.blur_size = max(3, int(blur_size) | 1)
        self.refresh_seconds = max(0.5, float(refresh_seconds))

        self._previous_gray: np.ndarray | None = None
        self._last_detection_time = 0.0
        self.last_motion_area = 0.0

    def should_run_detection(
        self,
        frame: np.ndarray,
    ) -> bool:
        now = time.monotonic()
        motion = self.detect_motion(frame)

        refresh_due = (
            now - self._last_detection_time
            >= self.refresh_seconds
        )

        if motion or refresh_due:
            self._last_detection_time = now
            return True

        return False

    def detect_motion(
        self,
        frame: np.ndarray,
    ) -> bool:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
        gray = cv2.GaussianBlur(
            gray,
            (self.blur_size, self.blur_size),
            0,
        )

        if self._previous_gray is None:
            self._previous_gray = gray
            self.last_motion_area = 0.0
            return True

        difference = cv2.absdiff(
            self._previous_gray,
            gray,
        )
        self._previous_gray = gray

        _, mask = cv2.threshold(
            difference,
            self.threshold,
            255,
            cv2.THRESH_BINARY,
        )
        mask = cv2.dilate(
            mask,
            None,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        self.last_motion_area = sum(
            cv2.contourArea(contour)
            for contour in contours
        )

        return self.last_motion_area >= self.minimum_area

    def reset(self) -> None:
        self._previous_gray = None
        self._last_detection_time = 0.0
        self.last_motion_area = 0.0
