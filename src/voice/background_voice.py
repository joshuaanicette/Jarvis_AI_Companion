import queue
import re
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.core.logger import logger


class BackgroundVoiceAssistant:
    def __init__(
        self,
        app,
        sample_rate: int = 16_000,
        channels: int = 1,
        microphone_device=None,
        silence_threshold: float = 450.0,
        silence_seconds: float = 1.2,
        minimum_phrase_seconds: float = 0.65,
        maximum_phrase_seconds: float = 20.0,
        block_seconds: float = 0.1,
        require_wake_word: bool = True,
        follow_up_window_seconds: float = 10.0,
    ):
        self.app = app
        self.sample_rate = sample_rate
        self.channels = channels
        self.microphone_device = microphone_device

        self.silence_threshold = silence_threshold
        self.silence_seconds = silence_seconds
        self.minimum_phrase_seconds = minimum_phrase_seconds
        self.maximum_phrase_seconds = maximum_phrase_seconds
        self.block_seconds = block_seconds

        self.require_wake_word = require_wake_word
        self.follow_up_window_seconds = follow_up_window_seconds

        self.running = False

        self._stop_event = threading.Event()
        self._audio_queue = queue.Queue()
        self._phrase_queue = queue.Queue()

        self._listener_thread = None
        self._processor_thread = None
        self._stream = None

        self._follow_up_deadline = 0.0

        self._temporary_directory = (
            Path(tempfile.gettempdir())
            / "jarvis_voice"
        )

        self._temporary_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def start(self) -> bool:
        if self.running:
            logger.info(
                "Background voice listener is already running"
            )
            return False

        self.running = True
        self._stop_event.clear()
        self.close_follow_up_window()

        self._clear_queue(
            self._audio_queue
        )

        self._clear_queue(
            self._phrase_queue
        )

        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="JarvisVoiceListener",
        )

        self._processor_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="JarvisVoiceProcessor",
        )

        self._listener_thread.start()
        self._processor_thread.start()

        logger.info(
            "Continuous background voice listener started"
        )

        return True

    def stop(self) -> None:
        logger.info(
            "Stopping background voice listener"
        )

        self._stop_event.set()
        self.running = False
        self.close_follow_up_window()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()

            except Exception as error:
                logger.warning(
                    "Could not close microphone stream: %s",
                    error,
                )

            self._stream = None

        self._phrase_queue.put(
            None
        )

        current_thread = threading.current_thread()

        if (
            self._listener_thread is not None
            and self._listener_thread.is_alive()
            and self._listener_thread is not current_thread
        ):
            self._listener_thread.join(
                timeout=2.0
            )

        if (
            self._processor_thread is not None
            and self._processor_thread.is_alive()
            and self._processor_thread is not current_thread
        ):
            self._processor_thread.join(
                timeout=2.0
            )

        logger.info(
            "Background voice listener stopped"
        )

    def open_follow_up_window(
        self,
        seconds: float | None = None,
    ) -> None:
        duration = (
            seconds
            if seconds is not None
            else self.follow_up_window_seconds
        )

        self._follow_up_deadline = (
            time.monotonic() + duration
        )

        logger.info(
            "Voice follow-up window opened for %.1f seconds",
            duration,
        )

    def close_follow_up_window(self) -> None:
        self._follow_up_deadline = 0.0

    def _follow_up_window_active(self) -> bool:
        return (
            self._follow_up_deadline > 0.0
            and time.monotonic()
            <= self._follow_up_deadline
        )

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ) -> None:
        if status:
            logger.warning(
                "Microphone stream status: %s",
                status,
            )

        if self._stop_event.is_set():
            return

        if self.app.speaking_event.is_set():
            return

        self._audio_queue.put(
            indata.copy()
        )

    def _listen_loop(self) -> None:
        block_size = max(
            1,
            int(
                self.sample_rate
                * self.block_seconds
            ),
        )

        phrase_blocks = []
        speech_started = False
        silence_duration = 0.0
        phrase_duration = 0.0

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=block_size,
                device=self.microphone_device,
                callback=self._audio_callback,
            )

            self._stream.start()

            print()
            print(
                "Jarvis is continuously listening."
            )

            if self.require_wake_word:
                print(
                    "Begin commands with 'Jarvis'."
                )

            print(
                "After each task, Jarvis will ask "
                "what you want to do next."
            )

            print(
                "Say 'Jarvis stop' to shut down."
            )
            print()

            while not self._stop_event.is_set():
                try:
                    block = self._audio_queue.get(
                        timeout=0.2
                    )

                except queue.Empty:
                    continue

                if self.app.speaking_event.is_set():
                    phrase_blocks.clear()
                    speech_started = False
                    silence_duration = 0.0
                    phrase_duration = 0.0
                    continue

                block_duration = (
                    len(block)
                    / self.sample_rate
                )

                volume = self._calculate_rms(
                    block
                )

                is_speech = (
                    volume
                    >= self.silence_threshold
                )

                if is_speech:
                    if not speech_started:
                        logger.info(
                            "Voice activity detected"
                        )

                    speech_started = True
                    silence_duration = 0.0

                    phrase_blocks.append(
                        block
                    )

                    phrase_duration += block_duration

                elif speech_started:
                    phrase_blocks.append(
                        block
                    )

                    phrase_duration += block_duration
                    silence_duration += block_duration

                if (
                    speech_started
                    and silence_duration
                    >= self.silence_seconds
                ):
                    self._finish_phrase(
                        phrase_blocks,
                        phrase_duration,
                    )

                    phrase_blocks = []
                    speech_started = False
                    silence_duration = 0.0
                    phrase_duration = 0.0

                elif (
                    speech_started
                    and phrase_duration
                    >= self.maximum_phrase_seconds
                ):
                    logger.info(
                        "Maximum phrase duration reached"
                    )

                    self._finish_phrase(
                        phrase_blocks,
                        phrase_duration,
                    )

                    phrase_blocks = []
                    speech_started = False
                    silence_duration = 0.0
                    phrase_duration = 0.0

        except Exception as error:
            logger.exception(
                "Continuous microphone listener failed: %s",
                error,
            )

        finally:
            self.running = False

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()

                except Exception:
                    pass

                self._stream = None

    def _finish_phrase(
        self,
        phrase_blocks: list[np.ndarray],
        phrase_duration: float,
    ) -> None:
        if (
            not phrase_blocks
            or phrase_duration
            < self.minimum_phrase_seconds
        ):
            return

        audio = np.concatenate(
            phrase_blocks,
            axis=0,
        )

        trailing_samples = int(
            self.sample_rate * 0.25
        )

        if len(audio) > trailing_samples:
            audio = audio[
                : len(audio) - trailing_samples
            ]

        logger.info(
            "Completed voice phrase: %.2f seconds",
            phrase_duration,
        )

        self._phrase_queue.put(
            audio
        )

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio = self._phrase_queue.get(
                    timeout=0.25
                )

            except queue.Empty:
                continue

            if audio is None:
                break

            audio_path = None

            try:
                audio_path = self._save_wav(
                    audio
                )

                text = (
                    self.app.stt.transcribe(
                        str(audio_path)
                    )
                    .strip()
                )

                if not text:
                    continue

                command = self._prepare_voice_command(
                    text
                )

                if command is None:
                    continue

                print(
                    f"You: {text}"
                )

                if command == "__wake_only__":
                    response = (
                        "Yes? What would you like me to do?"
                    )

                    print(
                        f"Jarvis: {response}"
                    )

                    self.app.speak(
                        response
                    )

                    self.open_follow_up_window()
                    continue

                if command == "__no_follow_up__":
                    response = (
                        "Okay. Say Jarvis when you need me."
                    )

                    print(
                        f"Jarvis: {response}"
                    )

                    self.app.speak(
                        response
                    )

                    self.close_follow_up_window()
                    continue

                if self._is_stop_joe_command(
                    command
                ):
                    print(
                        "Jarvis: Stopping."
                    )

                    self.app.state.online = False
                    self._stop_event.set()
                    self.running = False
                    break

                if self._is_cancel_command(
                    command
                ):
                    response = (
                        self.app.cancel_active_task()
                    )

                    print(
                        f"Jarvis: {response}"
                    )

                    self.app.speak(
                        response
                    )

                    self._ask_follow_up()
                    continue

                response = (
                    self.app.conversation.process(
                        command
                    )
                )

                if response:
                    print(
                        f"Jarvis: {response}"
                    )

                self._ask_follow_up()

            except Exception as error:
                logger.exception(
                    "Voice phrase processing failed: %s",
                    error,
                )

            finally:
                if audio_path is not None:
                    try:
                        audio_path.unlink(
                            missing_ok=True
                        )

                    except OSError:
                        pass

        self.running = False

    def _ask_follow_up(self) -> None:
        if self._stop_event.is_set():
            return

        follow_up = (
            "What would you like me to do next?"
        )

        print(
            f"Jarvis: {follow_up}"
        )

        self.app.speak(
            follow_up
        )

        self.open_follow_up_window()

    def _prepare_voice_command(
        self,
        text: str,
    ) -> str | None:
        normalized = self._normalize_command(
            text
        )

        if not normalized:
            return None

        if self._is_stop_joe_command(
            normalized
        ):
            return normalized

        if self._follow_up_window_active():
            self.close_follow_up_window()

            if self._is_no_follow_up_command(
                normalized
            ):
                return "__no_follow_up__"

            return normalized

        if not self.require_wake_word:
            return normalized

        wake_patterns = (
            r"^hey jarvis\b",
            r"^okay jarvis\b",
            r"^ok jarvis\b",
            r"^jarvis\b",
        )

        for pattern in wake_patterns:
            match = re.match(
                pattern,
                normalized,
            )

            if not match:
                continue

            command = normalized[
                match.end():
            ].strip()

            if not command:
                return "__wake_only__"

            return command

        logger.info(
            "Ignored speech without wake word: %s",
            text,
        )

        return None

    def _save_wav(
        self,
        audio: np.ndarray,
    ) -> Path:
        timestamp = time.time_ns()

        path = (
            self._temporary_directory
            / f"phrase_{timestamp}.wav"
        )

        with wave.open(
            str(path),
            "wb",
        ) as wav_file:
            wav_file.setnchannels(
                self.channels
            )

            wav_file.setsampwidth(
                2
            )

            wav_file.setframerate(
                self.sample_rate
            )

            wav_file.writeframes(
                audio.astype(
                    np.int16
                ).tobytes()
            )

        return path

    @staticmethod
    def _calculate_rms(
        audio: np.ndarray,
    ) -> float:
        if audio.size == 0:
            return 0.0

        samples = audio.astype(
            np.float32
        )

        return float(
            np.sqrt(
                np.mean(
                    np.square(samples)
                )
            )
        )

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
    def _is_stop_joe_command(
        cls,
        text: str,
    ) -> bool:
        normalized = cls._normalize_command(
            text
        )

        return normalized in {
            "jarvis stop",
            "jarvis shutdown",
            "jarvis shut down",
            "jarvis quit",
            "jarvis exit",
            "jarvis finish",
        }

    @classmethod
    def _is_cancel_command(
        cls,
        text: str,
    ) -> bool:
        normalized = cls._normalize_command(
            text
        )

        return normalized in {
            "cancel task",
            "cancel the task",
            "cancel current task",
            "cancel the current task",
            "stop current task",
            "stop the current task",
        }

    @classmethod
    def _is_no_follow_up_command(
        cls,
        text: str,
    ) -> bool:
        normalized = cls._normalize_command(
            text
        )

        return normalized in {
            "nothing",
            "nothing else",
            "no",
            "no thanks",
            "no thank you",
            "never mind",
            "nevermind",
            "that is all",
            "that's all",
            "im done",
            "i'm done",
        }

    @staticmethod
    def _clear_queue(
        target_queue: queue.Queue,
    ) -> None:
        while True:
            try:
                target_queue.get_nowait()

            except queue.Empty:
                break