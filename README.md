# Jay AI Companion

A modular Raspberry Pi 5 AI companion scaffold with:

- Ollama local LLM integration
- Long-term memory
- Tool routing
- Text-to-speech interface
- Speech-to-text interface
- Computer vision pipeline
- Robotics abstractions
- Automation and networking layers
- Tests and demos

## Quick start

```bash
python3 -m venv .ai
source .ai/bin/activate
pip install -r requirements.txt
pytest
python3 -m demos.conversation_demo
```

## Important

This scaffold is designed to merge into an existing `~/Projects/Jay` project.
Hardware-specific implementations such as GPIO pins, Hailo acceleration,
camera drivers, microphones, and motor drivers may require adjustment for
your exact parts and Ubuntu version.
