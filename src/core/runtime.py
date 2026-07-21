import re
import signal
import sys
import threading
import time
from enum import Enum

from src.core.application import Application
from src.core.logger import logger


class RuntimeMode(str, Enum):
    VOICE = "voice"
    TEXT = "text"
    HYBRID = "hybrid"
    ONCE = "once"


class JayRuntime:
    """
    Official Joe runtime controller.

    Modes:

    voice:
        Continuously listens through the microphone.

    text:
        Accepts typed terminal conversations.

    hybrid:
        Continuously listens while also accepting typed input.

    once:
        Processes one supplied command and exits.
    """

    def __init__(
        self,
        mode: RuntimeMode = RuntimeMode.VOICE,
    ):
        self.mode = mode
        self.app = Application()

        self.running = False

        self._shutdown_event = threading.Event()
        self._shutdown_complete = False

        self._text_thread: threading.Thread | None = None

        self._install_signal_handlers()

    def start(
        self,
        initial_command: str | None = None,
    ) -> int:
        logger.info(
            "Starting Joe runtime in %s mode",
            self.mode.value,
        )

        self.running = True
        self._shutdown_event.clear()
        self._shutdown_complete = False

        self.app.state.online = True

        self._print_startup_banner()

        try:
            if self.mode == RuntimeMode.ONCE:
                return self._run_once(
                    initial_command
                )

            if self.mode == RuntimeMode.VOICE:
                self._start_voice()
                self._wait_for_shutdown()
                return 0

            if self.mode == RuntimeMode.TEXT:
                self._run_text_conversation(
                    stop_runtime_when_finished=True
                )

                return 0

            if self.mode == RuntimeMode.HYBRID:
                self._start_voice()
                self._start_text_thread()
                self._wait_for_shutdown()

                return 0

            raise ValueError(
                f"Unsupported runtime mode: {self.mode}"
            )

        except KeyboardInterrupt:
            logger.info(
                "Keyboard interrupt received"
            )

            self.request_shutdown()
            return 0

        except Exception as error:
            logger.exception(
                "Joe runtime failed: %s",
                error,
            )

            print(
                f"Joe encountered an error: {error}",
                file=sys.stderr,
            )

            self.request_shutdown()
            return 1

        finally:
            self.shutdown()

    def _start_voice(self) -> None:
        logger.info(
            "Starting continuous microphone listener"
        )

        started = self.app.background_voice.start()

        if not started:
            raise RuntimeError(
                "The continuous voice listener "
                "could not be started."
            )

        logger.info(
            "Continuous voice mode started successfully"
        )

    def _run_text_conversation(
        self,
        stop_runtime_when_finished: bool,
    ) -> None:
        print()
        print("Text conversation is available.")
        print("Type 'joe stop' to shut down Joe.")
        print()

        while not self._shutdown_event.is_set():
            try:
                user_text = input(
                    "You: "
                ).strip()

            except EOFError:
                logger.info(
                    "Terminal input reached EOF"
                )
                break

            except KeyboardInterrupt:
                logger.info(
                    "Text input interrupted"
                )

                if stop_runtime_when_finished:
                    self.request_shutdown()

                break

            if not user_text:
                continue

            if self._is_shutdown_command(
                user_text
            ):
                print(
                    "Joe: Stopping."
                )

                self.request_shutdown()
                break

            try:
                response = (
                    self.app.conversation.process(
                        user_text
                    )
                )

                if response:
                    print(
                        f"Joe: {response}"
                    )

            except Exception as error:
                logger.exception(
                    "Text conversation failed: %s",
                    error,
                )

                print(
                    "Joe: I could not process that request."
                )

        if stop_runtime_when_finished:
            self.request_shutdown()

    def _start_text_thread(self) -> None:
        self._text_thread = threading.Thread(
            target=self._run_text_conversation,
            kwargs={
                "stop_runtime_when_finished": False,
            },
            daemon=True,
            name="JoeTextConversation",
        )

        self._text_thread.start()

        logger.info(
            "Hybrid text input thread started"
        )

    def _run_once(
        self,
        command: str | None,
    ) -> int:
        if not command:
            print(
                "A command is required when using once mode.",
                file=sys.stderr,
            )

            return 2

        response = self.app.conversation.process(
            command
        )

        if response:
            print(
                f"Joe: {response}"
            )

        return 0

    def _wait_for_shutdown(self) -> None:
        while not self._shutdown_event.is_set():
            if not self.app.state.online:
                logger.info(
                    "Application state changed to offline"
                )

                self.request_shutdown()
                break

            if self.mode in {
                RuntimeMode.VOICE,
                RuntimeMode.HYBRID,
            }:
                if not self.app.background_voice.running:
                    logger.info(
                        "Background voice listener stopped"
                    )

                    self.request_shutdown()
                    break

            time.sleep(
                0.25
            )

    def request_shutdown(self) -> None:
        if self._shutdown_event.is_set():
            return

        logger.info(
            "Runtime shutdown requested"
        )

        self._shutdown_event.set()
        self.running = False

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return

        self._shutdown_complete = True

        logger.info(
            "Stopping Joe runtime"
        )

        self.running = False
        self._shutdown_event.set()

        try:
            self.app.shutdown()

        except Exception as error:
            logger.exception(
                "Application shutdown failed: %s",
                error,
            )

        print()
        print("Joe has stopped.")

    def _install_signal_handlers(self) -> None:
        try:
            signal.signal(
                signal.SIGINT,
                self._handle_signal,
            )

            signal.signal(
                signal.SIGTERM,
                self._handle_signal,
            )

        except ValueError:
            logger.warning(
                "Signal handlers were not installed"
            )

    def _handle_signal(
        self,
        signum,
        frame,
    ) -> None:
        logger.info(
            "Received operating-system signal: %s",
            signum,
        )

        self.request_shutdown()

    def _print_startup_banner(self) -> None:
        print()
        print("=" * 58)
        print("Joe AI Companion")
        print("=" * 58)
        print(
            f"Runtime mode: {self.mode.value}"
        )

        if self.mode in {
            RuntimeMode.VOICE,
            RuntimeMode.HYBRID,
        }:
            print(
                "The microphone will listen continuously."
            )
            print(
                "Begin voice commands with 'Joe'."
            )
            print(
                "Say 'Joe stop' to shut down."
            )

        if self.mode in {
            RuntimeMode.TEXT,
            RuntimeMode.HYBRID,
        }:
            print(
                "Typed conversation is available."
            )
            print(
                "Type 'joe stop' to shut down."
            )

        print()
        print("Model routing:")
        print(
            "  Gemma: simple and conversational requests"
        )
        print(
            "  Qwen: engineering, math, science, history,"
        )
        print(
            "        programming, and complex reasoning"
        )
        print("=" * 58)
        print()

    @staticmethod
    def _normalize_command(
        text: str,
    ) -> str:
        normalized = text.lower().strip()

        normalized = re.sub(
            r"[^\w\s'-]",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @classmethod
    def _is_shutdown_command(
        cls,
        text: str,
    ) -> bool:
        normalized = cls._normalize_command(
            text
        )

        return normalized in {
            "joe stop",
            "joe shutdown",
            "joe shut down",
            "joe quit",
            "joe exit",
            "stop joe",
        }