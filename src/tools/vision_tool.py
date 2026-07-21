from src.core.logger import logger
from src.tools.tool import Tool


class VisionToolError(RuntimeError):
    """Raised when Jay cannot use the camera."""


class VisionTool(Tool):
    def __init__(self, vision_manager):
        self.vision_manager = vision_manager

    @property
    def name(self) -> str:
        return "vision"

    def execute(
        self,
        action: str = "scan",
        show_camera: bool = False,
    ) -> str:
        if action == "start":
            return self._start_continuous(show_camera=True)

        if action == "stop":
            return self._stop_continuous()

        if action == "status":
            return self._status()

        return self._scan_once(show_camera=show_camera)

    def _scan_once(self, show_camera: bool) -> str:
        try:
            self.vision_manager.start()

            frame = self.vision_manager.process_once(
                show_camera=show_camera,
            )

            return frame.scene

        except Exception as error:
            logger.exception("Vision scan failed: %s", error)

            raise VisionToolError(
                f"I could not use the camera: {error}"
            ) from error

        finally:
            self.vision_manager.stop()

    def _start_continuous(self, show_camera: bool) -> str:
        try:
            started = self.vision_manager.start_continuous(
                show_camera=show_camera,
                detection_interval=0.1,
            )

            if not started:
                return "The camera is already detecting objects."

            return (
                "I opened the camera and started continuous "
                "object detection. Say stop camera when you want me to stop."
            )

        except Exception as error:
            logger.exception(
                "Could not start continuous vision: %s",
                error,
            )

            raise VisionToolError(
                f"I could not start the camera: {error}"
            ) from error

    def _stop_continuous(self) -> str:
        stopped = self.vision_manager.stop_continuous()

        if not stopped:
            return "The camera is not currently running."

        return "I stopped the camera and object detection."

    def _status(self) -> str:
        if self.vision_manager.running:
            scene = self.vision_manager.latest_scene

            if scene:
                return (
                    "The camera is running. "
                    f"My latest detection is: {scene}"
                )

            return "The camera is running, but I have not detected anything yet."

        return "The camera is not currently running."