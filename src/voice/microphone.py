import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.core.logger import logger


class MicrophoneError(RuntimeError):
    """Raised when microphone recording fails."""


class Microphone:
    """
    Records fixed-duration mono audio and saves it as a WAV file.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

    @staticmethod
    def list_devices():
        return sd.query_devices()

    def record(
        self,
        duration: float = 5.0,
        output_path: str | Path = "data/audio/user_input.wav",
    ) -> Path:
        if duration <= 0:
            raise ValueError("Recording duration must be greater than zero.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        frame_count = int(duration * self.sample_rate)

        logger.info("Recording microphone for %.1f seconds", duration)

        try:
            audio = sd.rec(
                frame_count,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device,
            )
            sd.wait()
        except Exception as error:
            raise MicrophoneError(
                f"Microphone recording failed: {error}"
            ) from error

        self._save_wav(path, audio)

        logger.info("Saved microphone recording: %s", path)
        return path

    def _save_wav(self, path: Path, audio: np.ndarray) -> None:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # int16 = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())