from __future__ import annotations

from collections import Counter
from typing import Any


class SceneAnalyzer:
    """
    Convert detector output into a short spoken description.
    """

    def analyze(
        self,
        detections: Any,
    ) -> str:
        return self.describe(detections)

    def summarize(
        self,
        detections: Any,
    ) -> str:
        return self.describe(detections)

    def describe(
        self,
        frame_or_detections: Any,
    ) -> str:
        detections = self._extract_detections(
            frame_or_detections
        )

        labels = [
            self._get_label(detection)
            for detection in detections
        ]

        labels = [
            label
            for label in labels
            if label
        ]

        if not labels:
            return (
                "I opened the camera, but I did not "
                "recognize any objects."
            )

        return (
            "I can see "
            + self._format_labels(labels)
            + "."
        )

    @staticmethod
    def _extract_detections(
        value: Any,
    ) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, dict):
            for key in (
                "objects",
                "detections",
                "results",
            ):
                items = value.get(key)

                if isinstance(items, list):
                    return items

            return []

        for attribute in (
            "objects",
            "detections",
            "results",
        ):
            items = getattr(
                value,
                attribute,
                None,
            )

            if items is not None:
                try:
                    return list(items)
                except TypeError:
                    return []

        return []

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
                    return str(value).strip()

            return ""

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
                return str(value).strip()

        return ""

    @staticmethod
    def _format_labels(
        labels: list[str],
    ) -> str:
        counts = Counter(
            label.lower()
            for label in labels
            if label
        )

        formatted: list[str] = []

        for label, count in counts.items():
            if count == 1:
                article = (
                    "an"
                    if label[0].lower()
                    in {"a", "e", "i", "o", "u"}
                    else "a"
                )

                formatted.append(
                    f"{article} {label}"
                )
            else:
                plural = (
                    label
                    if label.endswith("s")
                    else f"{label}s"
                )

                formatted.append(
                    f"{count} {plural}"
                )

        if len(formatted) == 1:
            return formatted[0]

        if len(formatted) == 2:
            return (
                f"{formatted[0]} and {formatted[1]}"
            )

        return (
            ", ".join(formatted[:-1])
            + f", and {formatted[-1]}"
        )
