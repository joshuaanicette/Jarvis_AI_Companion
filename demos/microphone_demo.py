from src.voice.microphone import Microphone, MicrophoneError
from src.voice.whisper_stt import WhisperSpeechToText


def main() -> None:
    print("\nJay microphone test")
    print("-------------------")

    microphone = Microphone(
        sample_rate=16000,
        channels=1,
    )

    speech_to_text = WhisperSpeechToText(
        model_size="tiny.en",
        device="cpu",
        compute_type="int8",
    )

    input("Press Enter when you are ready to speak...")

    try:
        print("Recording now. Speak for 5 seconds.")

        audio_path = microphone.record(
            duration=5.0,
            output_path="data/audio/microphone_test.wav",
        )

        print(f"Recording saved to: {audio_path}")
        print("Transcribing audio...")

        text = speech_to_text.transcribe(str(audio_path))

        if text:
            print(f"You said: {text}")
        else:
            print("No speech was detected.")

    except MicrophoneError as error:
        print(f"Microphone error: {error}")

    except Exception as error:
        print(f"Unexpected error: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()