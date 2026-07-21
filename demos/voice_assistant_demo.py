from src.core.application import Application
from src.voice.microphone import MicrophoneError
from src.voice.piper_tts import PiperTextToSpeechError


def main():
    app = Application()

    print("\nJay voice mode")
    print("Press Enter to speak.")
    print("Type 'exit' and press Enter to stop.\n")

    while True:
        command = input(
            "Press Enter to record, or type exit: "
        ).strip()

        if command.lower() == "exit":
            print("Shutting down Jay...")
            break

        try:
            audio_path = app.microphone.record(
                duration=6,
                output_path="data/audio/user_input.wav",
            )
        except MicrophoneError as error:
            print(f"Microphone error: {error}")
            continue

        text = app.stt.transcribe(str(audio_path))

        if not text:
            print("Jay did not hear speech clearly.")
            continue

        print(f"You: {text}")

        response = app.conversation.process(text)
        print(f"Jay: {response}")


if __name__ == "__main__":
    main()