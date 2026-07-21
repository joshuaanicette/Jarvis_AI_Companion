import time

from src.core.application import Application


def main() -> None:
    app = Application()
    app.state.online = True

    print()
    print("Jay background voice mode")
    print("-------------------------")
    print("The microphone is continuously listening.")
    print("Say commands such as:")
    print("- What is the weather in Miami?")
    print("- Show me the weather dashboard for Miami.")
    print("- What should I wear outside?")
    print("- Start the camera and keep detecting objects.")
    print("- What are you detecting?")
    print("- Stop the camera.")
    print("- Cancel the current task.")
    print("- Use Qwen and explain Ohm's law.")
    print("- Use Gemma and tell me a joke.")
    print("- Jay stop.")
    print()

    started = app.background_voice.start()

    if not started:
        print("The background voice listener could not start.")
        return

    try:
        while app.background_voice.running:
            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        app.shutdown()
        print("Jay has stopped.")


if __name__ == "__main__":
    main()