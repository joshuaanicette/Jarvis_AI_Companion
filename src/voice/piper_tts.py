import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "voice"
    / "en_US-lessac-medium.onnx"
)

from src.core.logger import logger
from src.voice.text_to_speech import TextToSpeech


class PiperTextToSpeechError(RuntimeError):
    """Raised when Piper synthesis or playback fails."""


class PiperTextToSpeech(TextToSpeech):
    """
    Generates speech through the Piper CLI and plays it with ALSA.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        speaker_device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.speaker_device = speaker_device

        self.piper_command = shutil.which("piper")
        self.aplay_command = shutil.which("aplay")

        logger.info("Initializing Piper TTS")

    def speak(self, text: str) -> None:
        cleaned_text = text.strip()

        if not cleaned_text:
            return

        if not self.piper_command:
            raise PiperTextToSpeechError(
                "The piper command was not found."
            )

        if not self.aplay_command:
            raise PiperTextToSpeechError(
                "The aplay command was not found."
            )

        if not self.model_path.exists():
            raise PiperTextToSpeechError(
                f"Piper voice model not found: {self.model_path}"
            )

        logger.info("Piper says: %s", cleaned_text)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            output_path = Path(temporary_file.name)

        try:
            subprocess.run(
                [
                    self.piper_command,
                    "--model",
                    str(self.model_path),
                    "--output_file",
                    str(output_path),
                ],
                input=cleaned_text,
                text=True,
                check=True,
                capture_output=True,
            )

            playback_command = [
                self.aplay_command,
                "-q",
            ]

            if self.speaker_device:
                playback_command.extend(
                    ["-D", self.speaker_device]
                )

            playback_command.append(str(output_path))

            subprocess.run(
                playback_command,
                check=True,
            )

        except subprocess.CalledProcessError as error:
            raise PiperTextToSpeechError(
                f"Piper speech failed: {error}"
            ) from error

        finally:
            output_path.unlink(missing_ok=True)