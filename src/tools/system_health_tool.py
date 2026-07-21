from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.tools.tool import Tool


class SystemHealthTool(Tool):
    @property
    def name(self):
        return "system_health"

    def can_handle(self, text: str) -> bool:
        normalized = text.lower()
        phrases = (
            "system health",
            "system status",
            "check your system",
            "check the system",
            "cpu temperature",
            "cpu usage",
            "memory usage",
            "ram usage",
            "disk space",
            "how are your systems",
            "how is the raspberry pi",
        )
        return any(phrase in normalized for phrase in phrases)

    def execute(self, text: str = "") -> str:
        status = self.get_status()
        parts = [
            f"CPU usage is {status['cpu_percent']:.0f} percent",
            f"memory usage is {status['memory_percent']:.0f} percent",
            f"disk usage is {status['disk_percent']:.0f} percent",
        ]

        temperature = status.get("temperature_c")
        if temperature is not None:
            parts.append(
                f"CPU temperature is {temperature:.1f} degrees Celsius"
            )

        if status["ollama_running"]:
            parts.append("Ollama is running")
        else:
            parts.append("Ollama is not currently responding")

        return ". ".join(parts) + "."

    def run(self, text: str = "") -> str:
        return self.execute(text)

    def get_status(self) -> dict[str, Any]:
        memory_total, memory_available = self._memory_info()
        disk = shutil.disk_usage("/")

        memory_used = max(0, memory_total - memory_available)
        memory_percent = (
            memory_used / memory_total * 100
            if memory_total
            else 0.0
        )

        return {
            "cpu_percent": self._cpu_percent(),
            "temperature_c": self._temperature(),
            "memory_total_bytes": memory_total,
            "memory_available_bytes": memory_available,
            "memory_percent": memory_percent,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "disk_percent": (
                disk.used / disk.total * 100
                if disk.total
                else 0.0
            ),
            "load_average": os.getloadavg(),
            "ollama_running": self._ollama_running(),
        }

    @staticmethod
    def _cpu_percent() -> float:
        try:
            import psutil
            return float(psutil.cpu_percent(interval=0.25))
        except Exception:
            load_1m = os.getloadavg()[0]
            cpu_count = max(1, os.cpu_count() or 1)
            return min(100.0, load_1m / cpu_count * 100.0)

    @staticmethod
    def _memory_info() -> tuple[int, int]:
        try:
            import psutil
            memory = psutil.virtual_memory()
            return int(memory.total), int(memory.available)
        except Exception:
            values = {}
            with Path("/proc/meminfo").open(
                "r",
                encoding="utf-8",
            ) as file:
                for line in file:
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip().split()[0]) * 1024

            return (
                values.get("MemTotal", 0),
                values.get("MemAvailable", 0),
            )

    @staticmethod
    def _temperature() -> float | None:
        paths = (
            Path("/sys/class/thermal/thermal_zone0/temp"),
            Path("/sys/class/hwmon/hwmon0/temp1_input"),
        )

        for path in paths:
            try:
                raw = float(path.read_text().strip())
                return raw / 1000.0 if raw > 200 else raw
            except (OSError, ValueError):
                continue

        return None

    @staticmethod
    def _ollama_running() -> bool:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
