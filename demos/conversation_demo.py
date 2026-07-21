from src.core.application import Application


def main():
    app = Application()
    print("\nJay is ready! Type 'exit' to quit.\n")

    while True:
        text = input("You: ").strip()
        if text.lower() == "exit":
            print("Shutting down Jay...")
            break
        print(f"Jay: {app.conversation.process(text)}")


if __name__ == "__main__":
    main()
