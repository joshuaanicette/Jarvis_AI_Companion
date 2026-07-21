import threading
import time

from src.ai.memory import MemoryManager
from src.ai.memory_analyzer import MemoryAnalyzer
from src.ai.memory_retriever import MemoryRetriever
from src.ai.model_router import ModelRouter
from src.ai.ollama_llm import OllamaLLM
from src.ai.subject_router import SubjectRouter

from src.automation.productivity_manager import (
    ProductivityManager,
)

from src.core.config import Config
from src.core.event_bus import EventBus
from src.core.hardware import HardwareConfig
from src.core.logger import logger
from src.core.state import JayState

from src.location.location_manager import (
    LocationManager,
)

from src.navigation.dashboard import (
    NavigationDashboard,
)

from src.robotics.mock_motor_controller import (
    MockMotorController,
)
from src.robotics.robot import Robot

from src.tools.navigation_tool import (
    NavigationTool,
)
from src.tools.productivity_tool import (
    ProductivityTool,
)
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter
from src.tools.system_health_tool import (
    SystemHealthTool,
)
from src.tools.time_tool import TimeTool
from src.tools.vision_tool import VisionTool
from src.tools.weather_tool import WeatherTool

from src.vision.camera import Camera
from src.vision.detector import ObjectDetector
from src.vision.room_scanner import RoomScanner
from src.vision.scene_analyzer import (
    SceneAnalyzer,
)
from src.vision.vision_manager import (
    VisionManager,
)

from src.voice.background_voice import (
    BackgroundVoiceAssistant,
)
from src.voice.conversation import (
    ConversationManager,
)
from src.voice.microphone import Microphone
from src.voice.piper_tts import (
    PiperTextToSpeech,
    PiperTextToSpeechError,
)
from src.voice.whisper_stt import (
    WhisperSpeechToText,
)

from src.weather.clothing_advisor import (
    ClothingAdvisor,
)
from src.weather.dashboard import (
    WeatherDashboard,
)

from src.web.chat_dashboard import (
    JayChatDashboard,
)


class Application:
    def __init__(
        self,
    ):
        logger.info(
            "Building Jarvis Application"
        )

        # Core services
        self.config = Config()
        self.hardware = HardwareConfig()
        self.events = EventBus()
        self.state = JayState()

        # Prevent microphone capture while Jarvis speaks.
        self.speaking_event = threading.Event()

        # Language model configuration
        llm_config = self.config.get(
            "llm",
            {},
        )

        self.llm = OllamaLLM(
            host=llm_config.get(
                "host",
                "http://localhost:11434",
            ),
            model=llm_config.get(
                "model",
                "gemma3:1b",
            ),
        )

        self.model_router = ModelRouter(
            fast_model="gemma3:1b",
            reasoning_model="qwen2.5:3b",
        )

        # Memory
        self.memory = MemoryManager(
            memory_path=(
                "data/memory/memory.json"
            ),
        )

        self.memory_analyzer = MemoryAnalyzer(
            llm=self.llm,
            model="gemma3:1b",
        )

        self.memory_retriever = MemoryRetriever(
            memory_manager=self.memory,
            max_results=8,
        )

        self.subject_router = SubjectRouter()

        # Voice output
        self.tts = PiperTextToSpeech()

        # Voice input
        self.microphone = Microphone(
            sample_rate=16_000,
            channels=1,
        )

        self.stt = WhisperSpeechToText(
            model_size="tiny.en",
            device="cpu",
            compute_type="int8",
        )

        # Vision configuration
        vision_config = self.config.get(
            "vision",
            {},
        )

        camera_width = vision_config.get(
            "width",
            640,
        )

        camera_height = vision_config.get(
            "height",
            480,
        )

        camera_fps = vision_config.get(
            "fps",
            10,
        )

        detection_confidence = (
            vision_config.get(
                "confidence",
                0.45,
            )
        )

        # Camera
        self.camera = Camera(
            index="/dev/video0",
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            backend="opencv",
            auto_detect=False,
        )

        # Object detection
        self.object_detector = ObjectDetector(
            model_name=vision_config.get(
                "model",
                "yolo11n.pt",
            ),
            confidence=detection_confidence,
        )

        self.scene_analyzer = SceneAnalyzer()

        # Room scanning
        self.room_scanner = RoomScanner(
            frame_width=camera_width,
            frame_height=camera_height,
            focal_length_x_pixels=700.0,
            focal_length_y_pixels=700.0,
            minimum_confidence=(
                detection_confidence
            ),
        )

        self.vision = VisionManager(
            camera=self.camera,
            detector=self.object_detector,
            analyzer=self.scene_analyzer,
            room_scanner=self.room_scanner,
        )

        # Weather
        self.weather_tool = WeatherTool()
        self.clothing_advisor = (
            ClothingAdvisor()
        )

        self.weather_dashboard = (
            WeatherDashboard(
                weather_tool=self.weather_tool,
                host="127.0.0.1",
                port=8765,
            )
        )

        # Navigation
        self.location_manager = (
            LocationManager()
        )

        self.navigation_dashboard = (
            NavigationDashboard(
                location_manager=(
                    self.location_manager
                ),
                host="127.0.0.1",
                port=8770,
            )
        )

        # Persistent reminders
        self.productivity = (
            ProductivityManager(
                database_path=(
                    "data/productivity/"
                    "productivity.db"
                ),
                poll_interval_seconds=5.0,
            )
        )

        # Tools
        self.tools = ToolRegistry()

        self.tools.register(
            TimeTool()
        )

        self.tools.register(
            SystemHealthTool()
        )

        self.tools.register(
            ProductivityTool(
                self.productivity
            )
        )

        self.tools.register(
            self.weather_tool
        )

        self.tools.register(
            VisionTool(
                self.vision
            )
        )

        self.tools.register(
            NavigationTool(
                dashboard=self.navigation_dashboard,
                location_manager=self.location_manager
            )
        )

        self.tool_router = ToolRouter(
            self.tools,
            app=self,
        )

        # Robotics
        self.robot = Robot(
            MockMotorController()
        )

        # Conversation manager
        self.conversation = (
            ConversationManager(
                self
            )
        )

        # Shared browser dashboard.
        # The terminal and browser use the same
        # Application instance.
        self.chat_dashboard = (
            JayChatDashboard(
                application=self,
                host="127.0.0.1",
                port=8780,
            )
        )

        try:
            self.chat_dashboard.start()

        except Exception as error:
            logger.exception(
                (
                    "Chat dashboard startup "
                    "failed: %s"
                ),
                error,
            )

        # Continuous voice assistant
        self.background_voice = (
            BackgroundVoiceAssistant(
                app=self,
                sample_rate=16_000,
                channels=1,
                microphone_device=None,
                silence_threshold=450.0,
                silence_seconds=1.2,
                minimum_phrase_seconds=0.65,
                maximum_phrase_seconds=20.0,
                block_seconds=0.1,
                require_wake_word=True,
                follow_up_window_seconds=10.0,
            )
        )

        # Start reminder monitoring
        self.productivity.start(
            callback=self._deliver_reminder
        )

        logger.info(
            (
                "Jarvis application built "
                "successfully"
            )
        )

    def speak(
        self,
        text: str,
    ) -> None:
        """
        Speak through Piper while temporarily
        disabling microphone capture.
        """

        cleaned_text = str(
            text
        ).strip()

        if not cleaned_text:
            return

        self.speaking_event.set()
        self.state.speaking = True

        try:
            self.tts.speak(
                cleaned_text
            )

        except PiperTextToSpeechError as error:
            logger.error(
                "TTS playback failed: %s",
                error,
            )

        except Exception as error:
            logger.exception(
                "Unexpected TTS error: %s",
                error,
            )

        finally:
            time.sleep(
                0.75
            )

            self.state.speaking = False
            self.speaking_event.clear()

    def _deliver_reminder(
        self,
        message: str,
    ) -> None:
        """
        Deliver reminders through the terminal
        and Piper speech.
        """

        cleaned_message = str(
            message
        ).strip()

        if not cleaned_message:
            return

        logger.info(
            "Delivering reminder: %s",
            cleaned_message,
        )

        print(
            f"\nJarvis: {cleaned_message}",
            flush=True,
        )

        self.speak(
            cleaned_message
        )

    def cancel_active_task(
        self,
    ) -> str:
        """
        Cancel the currently running task without
        shutting down Jarvis.
        """

        if self.vision.running:
            self.vision.stop_continuous()

            return (
                "I cancelled the active camera "
                "and object-detection task."
            )

        if self.vision.started:
            self.vision.stop()

            return (
                "I stopped the active camera task."
            )

        return (
            "There is no active background "
            "task to cancel."
        )

    def stop_active_tasks(
        self,
    ) -> None:
        """
        Stop background tools without shutting
        down Jarvis.
        """

        logger.info(
            "Stopping active tasks"
        )

        if self.vision.running:
            self.vision.stop_continuous()

        elif self.vision.started:
            self.vision.stop()

    def shutdown(
        self,
    ) -> None:
        """
        Shut down every Jarvis subsystem.
        """

        logger.info(
            "Shutting down Jarvis"
        )

        self.state.online = False

        try:
            if self.background_voice.running:
                self.background_voice.stop()

        except Exception as error:
            logger.exception(
                (
                    "Background voice shutdown "
                    "failed: %s"
                ),
                error,
            )

        try:
            self.productivity.stop()

        except Exception as error:
            logger.exception(
                (
                    "Productivity shutdown "
                    "failed: %s"
                ),
                error,
            )

        try:
            if self.vision.running:
                self.vision.stop_continuous()

            elif self.vision.started:
                self.vision.stop()

        except Exception as error:
            logger.exception(
                "Vision shutdown failed: %s",
                error,
            )

        try:
            self.chat_dashboard.stop()

        except Exception as error:
            logger.exception(
                (
                    "Chat dashboard shutdown "
                    "failed: %s"
                ),
                error,
            )

        try:
            self.navigation_dashboard.stop()

        except Exception as error:
            logger.exception(
                (
                    "Navigation dashboard "
                    "shutdown failed: %s"
                ),
                error,
            )

        try:
            self.weather_dashboard.stop()

        except Exception as error:
            logger.exception(
                (
                    "Weather dashboard shutdown "
                    "failed: %s"
                ),
                error,
            )

        logger.info(
            "Jarvis shutdown completed"
        )