# Jarvis — Local AI Companion

A modular, multimodal AI assistant built in Python for the Raspberry Pi 5. Jarvis runs local language models, processes voice and camera input, displays interactive web dashboards, plans routes, retrieves weather information, and manages persistent reminders through a unified software architecture.

Jarvis is designed as both a practical personal assistant and an expandable robotics platform. Its subsystem-based architecture allows vision, speech, navigation, weather, memory, automation, and AI reasoning services to operate independently while sharing one central application runtime.

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Technology Stack](#technology-stack)
* [Hardware](#hardware)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running Jarvis](#running-jarvis)
* [Interaction Modes](#interaction-modes)
* [Example Commands](#example-commands)
* [Subsystems](#subsystems)
* [Database Storage](#database-storage)
* [Testing](#testing)
* [Troubleshooting](#troubleshooting)
* [Security and Privacy](#security-and-privacy)
* [Roadmap](#roadmap)
* [Skills Demonstrated](#skills-demonstrated)
* [License](#license)

---

## Overview

Jarvis is a locally hosted AI companion that combines language-model reasoning with real-world tools and hardware interfaces.

The assistant can:

* Route general and technical prompts between multiple local AI models
* Understand typed and spoken commands
* Respond through synthesized speech
* Detect and analyze objects through a USB camera
* Display weather, navigation, camera, and chat information in browser dashboards
* Store reminders, tasks, memories, and conversation data locally
* Run background services using multithreading
* Support future integration with motors, sensors, and a mobile rover chassis

The project is built around a shared `Application` instance that initializes and coordinates every subsystem. This prevents individual services from creating duplicate AI models, cameras, microphones, databases, or web servers.

---

## Key Features

### Local AI Model Routing

Jarvis routes requests between local Ollama models based on the type and complexity of the prompt.

* **Gemma** handles general conversation and lightweight requests
* **Qwen** handles mathematics, programming, engineering, physics, science, and technical reasoning
* Model selection is managed through a centralized routing layer
* Prompts and responses can remain on the Raspberry Pi

### Voice Interaction

* Speech recognition using Whisper
* Text-to-speech output using Piper
* Wake-word-based background listening
* Follow-up conversation window
* Browser microphone recording
* Voice output toggle
* Microphone capture pauses while Jarvis is speaking to reduce feedback

### Computer Vision

* USB camera input through OpenCV
* YOLO object detection
* Confidence scoring
* Annotated frame previews
* Scene interpretation
* Camera snapshots through the browser dashboard
* Continuous or single-frame vision processing
* Expandable room-scanning and object-measurement support

### Weather

* Current weather and forecast retrieval
* FastAPI weather dashboard
* Browser-based forecast display
* Clothing recommendations based on weather conditions
* REST API integration

### Navigation

* Route planning between locations
* Current-location support through browser geolocation
* Driving, walking, and transit comparisons
* Distance and estimated travel time
* Interactive map dashboard
* Google Maps or other routing-service integration

### Productivity and Reminders

* Natural-language reminder creation
* Persistent SQLite storage
* Reminder listing
* Reminder editing and rescheduling
* Reminder renaming
* Reminder deletion
* Background reminder monitoring
* Real-time spoken alerts
* Persistent task management

### Web Dashboard

* Chat-style browser interface
* Model and category labels
* Weather and navigation shortcuts
* Camera preview panel
* Voice input controls
* Light and dark themes
* Responsive layout
* FastAPI backend with HTML, CSS, and JavaScript frontend

### Memory and Persistence

* Confirmed user memories
* Interest detection and reinforcement
* Context retrieval for relevant prompts
* JSON-based personal memory
* SQLite-based reminders and tasks
* Expandable SQLite conversation history

---

## System Architecture

```text
                         ┌─────────────────────────┐
                         │       User Input        │
                         │ Text • Voice • Browser  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Jarvis Runtime      │
                         │   Shared Application    │
                         └────────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
                ▼                     ▼                     ▼
       ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
       │ Subject Router │   │  Tool Router   │   │ Memory System  │
       └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
               │                    │                    │
               ▼                    ▼                    ▼
       ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
       │  Model Router  │   │ Registered     │   │ Relevant User  │
       │ Gemma / Qwen   │   │ Tools          │   │ Context        │
       └───────┬────────┘   └───────┬────────┘   └────────────────┘
               │                    │
               ▼                    ├──────── Weather
       ┌────────────────┐           ├──────── Navigation
       │     Ollama     │           ├──────── Vision
       │ Local Inference│           ├──────── Reminders
       └───────┬────────┘           └──────── System Health
               │
               ▼
       ┌────────────────┐
       │ Response Output│
       │ Text • Piper   │
       └────────────────┘
```

### Design Principles

* **Modularity:** Each subsystem is isolated behind a clear interface
* **Single ownership:** Hardware and services are initialized once
* **Local-first operation:** AI inference and persistent data remain local whenever possible
* **Extensibility:** Additional tools can be registered without rewriting the main runtime
* **Concurrency:** Background voice, reminders, dashboards, and monitoring services run independently
* **Fault isolation:** A subsystem failure should not terminate the entire assistant

---

## Technology Stack

### Backend

* Python 3
* FastAPI
* Uvicorn
* SQLite
* Multithreading
* REST APIs
* Pydantic
* Requests
* Python dotenv

### Artificial Intelligence

* Ollama
* Gemma
* Qwen
* Whisper
* Piper TTS
* YOLO

### Computer Vision

* OpenCV
* Ultralytics YOLO
* NumPy
* USB camera input

### Frontend

* HTML5
* CSS3
* JavaScript
* Fetch API
* MediaRecorder API
* Browser Geolocation API
* Leaflet or Google Maps integration

### Development and Testing

* Pytest
* Python virtual environments
* Git
* GitHub
* VS Code
* Linux
* Raspberry Pi OS or Ubuntu ARM64

---

## Hardware

### Current Hardware

* Raspberry Pi 5
* 8 GB RAM
* Raspberry Pi AI HAT+
* USB camera
* USB microphone
* USB speaker or mini soundbar
* 128 GB microSD card
* Active cooling
* Raspberry Pi power supply

### Planned Rover Hardware

* Four-wheel robot chassis
* TT motors
* Motor driver
* Teensy or Arduino-compatible microcontroller
* 7.4 V rechargeable battery
* Step-down buck converter
* Power switches
* Distance sensors
* Encoders
* Additional cameras or sensors

> Motor and high-current devices should not be powered directly from Raspberry Pi GPIO pins.

---

## Project Structure

```text
Jarvis_AI_Companion/
├── config/
│   ├── settings.yaml
│   └── hardware.yaml
│
├── data/
│   ├── conversations/
│   ├── memory/
│   │   └── memory.json
│   ├── productivity/
│   │   └── productivity.db
│   └── sensor_logs/
│
├── hardware/
│   ├── datasheets/
│   ├── diagrams/
│   └── wiring/
│
├── logs/
│
├── models/
│   ├── llm/
│   ├── vision/
│   └── voice/
│
├── scripts/
│
├── src/
│   ├── ai/
│   │   ├── memory.py
│   │   ├── memory_analyzer.py
│   │   ├── memory_retriever.py
│   │   ├── model_router.py
│   │   ├── ollama_llm.py
│   │   └── subject_router.py
│   │
│   ├── automation/
│   │   └── productivity_manager.py
│   │
│   ├── core/
│   │   ├── application.py
│   │   ├── config.py
│   │   ├── event_bus.py
│   │   ├── events.py
│   │   ├── hardware.py
│   │   ├── logger.py
│   │   ├── runtime.py
│   │   └── state.py
│   │
│   ├── location/
│   │   └── location_manager.py
│   │
│   ├── navigation/
│   │   └── dashboard.py
│   │
│   ├── robotics/
│   │   ├── mock_motor_controller.py
│   │   └── robot.py
│   │
│   ├── tools/
│   │   ├── navigation_tool.py
│   │   ├── productivity_tool.py
│   │   ├── registry.py
│   │   ├── router.py
│   │   ├── system_health_tool.py
│   │   ├── time_tool.py
│   │   ├── vision_tool.py
│   │   └── weather_tool.py
│   │
│   ├── vision/
│   │   ├── camera.py
│   │   ├── detector.py
│   │   ├── room_scanner.py
│   │   ├── scene_analyzer.py
│   │   └── vision_manager.py
│   │
│   ├── voice/
│   │   ├── background_voice.py
│   │   ├── conversation.py
│   │   ├── microphone.py
│   │   ├── piper_tts.py
│   │   └── whisper_stt.py
│   │
│   ├── weather/
│   │   ├── clothing_advisor.py
│   │   └── dashboard.py
│   │
│   └── web/
│       ├── chat_dashboard.py
│       └── chat_models.py
│
├── static/
│   ├── chat/
│   │   ├── chat.css
│   │   └── chat.js
│   └── navigation/
│
├── templates/
│   ├── jarvis_chat.html
│   ├── navigation.html
│   └── weather.html
│
├── tests/
│   ├── test_config.py
│   ├── test_conversation.py
│   ├── test_core.py
│   ├── test_productivity.py
│   └── test_vision.py
│
├── .env
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Jarvis_AI_Companion.git
cd Jarvis_AI_Companion
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Update the System

```bash
sudo apt update
sudo apt upgrade -y
```

### 3. Install System Dependencies

```bash
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ffmpeg \
    sqlite3 \
    portaudio19-dev \
    libopenblas-dev \
    libjpeg-dev \
    libopencv-dev
```

### 4. Create a Python Virtual Environment

```bash
python3 -m venv .ai
source .ai/bin/activate
```

### 5. Upgrade Python Packaging Tools

```bash
python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel
```

### 6. Install Python Dependencies

```bash
pip install -r requirements.txt
```

A representative `requirements.txt` may include:

```text
fastapi
uvicorn
pydantic
requests
python-dotenv
PyYAML
opencv-python
numpy
ultralytics
faster-whisper
sounddevice
python-multipart
pytest
```

Some Raspberry Pi installations may require platform-specific package versions.

### 7. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
```

### 8. Download Local Models

```bash
ollama pull gemma3:1b
ollama pull qwen2.5:3b
```

Confirm installed models:

```bash
ollama list
```

### 9. Verify Ollama

```bash
ollama run gemma3:1b
```

Exit with:

```text
/bye
```

### 10. Install Piper Voice Assets

Place the Piper model and configuration files in the expected voice-model directory.

Example:

```text
models/voice/
├── en_US-lessac-medium.onnx
└── en_US-lessac-medium.onnx.json
```

Update the Piper configuration in the project when using a different voice model.

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
nano .env
```

Example:

```env
WEATHER_API_KEY=your_weather_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

OLLAMA_HOST=http://localhost:11434
FAST_MODEL=gemma3:1b
REASONING_MODEL=qwen2.5:3b
```

Do not commit `.env` to GitHub.

Add it to `.gitignore`:

```gitignore
.env
.ai/
__pycache__/
*.pyc
*.db
*.db-shm
*.db-wal
logs/
models/**/*.onnx
models/**/*.pt
```

### Application Settings

Example `config/settings.yaml`:

```yaml
assistant:
  name: Jarvis

llm:
  host: http://localhost:11434
  model: gemma3:1b
  reasoning_model: qwen2.5:3b

vision:
  width: 640
  height: 480
  fps: 10
  model: yolo11n.pt
  confidence: 0.45

voice:
  whisper_model: tiny.en
  sample_rate: 16000
  channels: 1
  wake_word: jarvis

weather:
  dashboard_port: 8765

navigation:
  dashboard_port: 8770

chat:
  dashboard_port: 8780

reminders:
  poll_interval_seconds: 5
```

### Hardware Configuration

Example `config/hardware.yaml`:

```yaml
camera:
  device: /dev/video0
  backend: opencv

microphone:
  device: null
  sample_rate: 16000

speaker:
  output_device: default

robot:
  enabled: false
```

---

## Running Jarvis

Activate the virtual environment:

```bash
cd /home/joshuaanicette/Jarvis_AI_Companion
source .ai/bin/activate
```

### Default Mode

```bash
python3 main.py
```

### Voice Mode

```bash
python3 main.py --mode voice
```

### Text Mode

```bash
python3 main.py --mode text
```

### Hybrid Mode

```bash
python3 main.py --mode hybrid
```

### Process One Command

```bash
python3 main.py \
    --mode once \
    --command "What time is it?"
```

### Open the Browser Dashboard

```text
http://127.0.0.1:8780
```

Weather dashboard:

```text
http://127.0.0.1:8765
```

Navigation dashboard:

```text
http://127.0.0.1:8770
```

To open the dashboard from another device on the same network, configure the FastAPI host as:

```python
host="0.0.0.0"
```

Then browse to:

```text
http://RASPBERRY_PI_IP:8780
```

Do not expose the dashboard directly to the public internet without authentication, HTTPS, and network controls.

---

## Interaction Modes

### Text Mode

The user types commands into the terminal.

```text
You: Explain how a buck converter works.
Jarvis: A buck converter is a DC-to-DC switching regulator...
```

### Voice Mode

Jarvis listens for its wake word and processes spoken commands.

```text
Jarvis, what is the weather today?
```

### Hybrid Mode

Text input, browser chat, voice recognition, speech output, dashboards, and background services operate together.

### One-Shot Mode

One command is processed, and the runtime exits afterward.

---

## Example Commands

### General AI

```text
Explain the difference between computer engineering and computer science.
```

```text
Summarize how a Raspberry Pi communicates with a microcontroller.
```

### Mathematics and Engineering

```text
Solve the integral of 3x squared plus 14x plus 6.
```

```text
Explain Kirchhoff's voltage law with an example circuit.
```

```text
Calculate the current through a 220-ohm resistor connected to 5 volts.
```

### Vision

```text
What can you see?
```

```text
Detect objects with the camera.
```

```text
Describe the current scene.
```

### Weather

```text
What is the weather today?
```

```text
What should I wear based on the forecast?
```

```text
Open the weather dashboard.
```

### Navigation

```text
How long does it take to drive to Drexel University?
```

```text
Compare driving, walking, and transit routes to the Philadelphia Museum of Art.
```

```text
Open the navigation dashboard.
```

### Reminders

```text
Remind me to test the camera tomorrow at 3 PM.
```

```text
Show my reminders.
```

```text
Change reminder 2 to finish the report tomorrow at 6 PM.
```

```text
Reschedule reminder 2 tomorrow at 8 PM.
```

```text
Rename reminder 2 to submit the final project.
```

```text
Delete reminder 2.
```

### Tasks

```text
Create a task to test the motor controller.
```

```text
Create a task to finish the circuit diagram with high priority.
```

```text
Show my tasks.
```

```text
Complete task 3.
```

---

## Subsystems

### Application Runtime

`Application` creates and owns all shared resources:

* AI models
* Memory
* Camera
* Object detector
* Speech recognition
* Text-to-speech
* Weather services
* Navigation services
* Reminder manager
* Tool registry
* Web dashboards
* Robotics interfaces

This prevents resource conflicts caused by creating multiple camera, microphone, database, or server instances.

### Model Router

The model router selects the most suitable model for each request.

```text
General conversation → Gemma
Engineering or math → Qwen
Tool request → Registered tool
```

### Tool Registry

Tools are registered centrally:

```python
self.tools.register(TimeTool())
self.tools.register(SystemHealthTool())
self.tools.register(ProductivityTool(self.productivity))
self.tools.register(self.weather_tool)
self.tools.register(VisionTool(self.vision))
self.tools.register(
    NavigationTool(
        dashboard=self.navigation_dashboard,
        location_manager=self.location_manager,
    )
)
```

### Event Bus

The event bus allows subsystems to communicate without becoming tightly coupled.

Potential events include:

```text
REMINDER_DUE
VISION_OBJECT_DETECTED
WEATHER_UPDATED
ROUTE_PLANNED
VOICE_COMMAND_RECEIVED
SYSTEM_SHUTDOWN
```

### Background Services

Threaded background services may include:

* Wake-word listening
* Reminder monitoring
* FastAPI dashboards
* Camera processing
* Robotics telemetry
* Sensor monitoring

Shared state should be protected with locks, events, or thread-safe queues.

---

## Database Storage

### Productivity Database

```text
data/productivity/productivity.db
```

Stores:

* Tasks
* Priorities
* Completion states
* Reminder messages
* Reminder times
* Delivery states

Inspect it with:

```bash
sqlite3 data/productivity/productivity.db
```

```sql
.tables
SELECT * FROM tasks;
SELECT * FROM reminders;
.quit
```

### Conversation Database

When persistent browser conversation history is enabled:

```text
data/conversations/conversations.db
```

Recommended schema:

```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    category TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE
);
```

### Memory File

```text
data/memory/memory.json
```

Stores confirmed memories and inferred user interests separately.

---

## Testing

Run all tests from the project root:

```bash
source .ai/bin/activate
pytest
```

Run with detailed output:

```bash
pytest -v
```

Run one test module:

```bash
pytest tests/test_conversation.py -v
```

Run tests with import path explicitly configured:

```bash
PYTHONPATH=. pytest -v
```

### Compile Important Files

```bash
python3 -m py_compile \
    main.py \
    src/core/application.py \
    src/core/runtime.py \
    src/voice/conversation.py \
    src/web/chat_dashboard.py \
    src/tools/productivity_tool.py
```

### Suggested Test Coverage

* Configuration loading
* Model selection
* Tool routing
* Reminder creation and editing
* Database persistence
* Conversation history separation
* Camera availability
* Vision failure handling
* Weather API failures
* Navigation API failures
* TTS exception handling
* Runtime startup and shutdown

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run the command from the project root:

```bash
cd /home/joshuaanicette/Jarvis_AI_Companion
PYTHONPATH=. python3 main.py
```

For tests:

```bash
PYTHONPATH=. pytest
```

Make sure package directories include:

```text
__init__.py
```

### Ollama Model Not Found

Check installed models:

```bash
ollama list
```

Download missing models:

```bash
ollama pull gemma3:1b
ollama pull qwen2.5:3b
```

Check the service:

```bash
systemctl status ollama
```

Restart it:

```bash
sudo systemctl restart ollama
```

### Raspberry Pi Freezes During AI Inference

Possible causes:

* Too many applications open
* Multiple Ollama requests running simultaneously
* Camera and Qwen inference competing for memory
* Large model size
* Browser, VS Code, and AI services exhausting available RAM

Recommended actions:

```bash
free -h
htop
```

Use smaller quantized models, close unused applications, and pause vision processing during heavier inference.

### Camera Not Found

List video devices:

```bash
ls -l /dev/video*
```

Check camera information:

```bash
v4l2-ctl --list-devices
```

Test with OpenCV:

```python
import cv2

camera = cv2.VideoCapture("/dev/video0")

if not camera.isOpened():
    raise RuntimeError("Camera could not be opened.")

success, frame = camera.read()
print(success, frame.shape if success else None)

camera.release()
```

### Camera Device Busy

Check the process using the camera:

```bash
fuser /dev/video0
```

Terminate the conflicting process only after confirming it is safe:

```bash
kill PROCESS_ID
```

Only one service should own the camera at a time.

### Whisper Does Not Detect Speech

Check microphone devices:

```bash
arecord -l
```

Record a test:

```bash
arecord \
    -D default \
    -f S16_LE \
    -r 16000 \
    -c 1 \
    test.wav
```

Play it:

```bash
aplay test.wav
```

Verify that FFmpeg is installed:

```bash
ffmpeg -version
```

### Piper Does Not Speak

Check the voice model paths and verify audio output:

```bash
speaker-test -t wav
```

Test Piper separately using the configured model.

### Weather API Failure

Confirm `.env` contains:

```env
WEATHER_API_KEY=your_key
```

Load environment variables from the project root and verify the API request independently.

### Navigation API Failure

Confirm:

```env
GOOGLE_MAPS_API_KEY=your_key
```

Verify that the required Maps APIs are enabled for the key.

### Dashboard Port Already in Use

Check the port:

```bash
sudo lsof -i :8780
```

Terminate the stale process or configure another port.

### SQLite Database Locked

Avoid creating multiple `ProductivityManager` or `ConversationStore` instances.

Check for duplicate Jarvis processes:

```bash
ps aux | grep python
```

Use one shared application instance.

### Template Missing

Ensure the dashboard expects the same filename that exists in `templates/`.

Example:

```python
TEMPLATE_PATH = (
    PROJECT_ROOT
    / "templates"
    / "jarvis_chat.html"
)
```

---

## Security and Privacy

Jarvis is designed as a local-first assistant, but some tools may use external APIs.

### Local Data

The following data can remain on the Raspberry Pi:

* Ollama prompts and responses
* Memories
* Reminder records
* Task records
* Conversation history
* Camera processing
* Whisper transcription
* Piper speech generation

### External Services

Depending on configuration, these features may send requests externally:

* Weather API
* Google Maps or routing API
* OpenStreetMap tile requests
* OSRM routing requests

### Recommended Practices

* Never commit `.env`
* Restrict API keys by service and IP when possible
* Do not expose FastAPI ports directly to the internet
* Use firewall rules
* Avoid storing sensitive camera images unnecessarily
* Validate uploaded audio and file sizes
* Use authentication before remote deployment
* Back up databases securely

---

## Roadmap

### Phase 1 — Core Assistant

* [x] Local Ollama integration
* [x] Gemma and Qwen model routing
* [x] Modular application structure
* [x] Tool registry and router
* [x] Persistent memory

### Phase 2 — Voice and Vision

* [x] Whisper speech recognition
* [x] Piper text-to-speech
* [x] Wake-word listening
* [x] USB camera support
* [x] YOLO object detection
* [x] Annotated previews
* [x] Scene analysis

### Phase 3 — Tools and Dashboards

* [x] Weather tool
* [x] Clothing advisor
* [x] Weather dashboard
* [x] Navigation dashboard
* [x] Browser chat interface
* [x] Reminder and task system
* [x] Reminder editing and rescheduling

### Phase 4 — Persistent Conversations

* [ ] ChatGPT-style recent conversation sidebar
* [ ] Separate context for each conversation
* [ ] SQLite conversation history
* [ ] Rename and delete conversations
* [ ] Automatic conversation titles
* [ ] Conversation search
* [ ] Conversation export

### Phase 5 — Robotics

* [ ] Motor-controller integration
* [ ] Rover movement commands
* [ ] Obstacle avoidance
* [ ] Autonomous room navigation
* [ ] Sensor fusion
* [ ] Object-following mode
* [ ] Battery monitoring
* [ ] Emergency shutdown circuit

### Phase 6 — Advanced Intelligence

* [ ] Retrieval-augmented generation
* [ ] Document ingestion
* [ ] Long-term semantic memory
* [ ] User authentication
* [ ] Plugin framework
* [ ] Offline route caching
* [ ] Home automation
* [ ] Multi-agent task coordination

---

## Skills Demonstrated

This project demonstrates experience with:

* Python software architecture
* Embedded Linux
* Raspberry Pi development
* Local AI deployment
* Large language models
* Model routing
* Prompt engineering
* Computer vision
* OpenCV
* YOLO
* Speech recognition
* Text-to-speech
* REST API integration
* FastAPI
* JavaScript
* HTML and CSS
* SQLite
* Database design
* Multithreading
* Natural-language parsing
* Hardware abstraction
* Robotics architecture
* Unit testing
* Debugging
* Git and GitHub

---

## Resume Summary

> Developed a modular Python AI assistant on a Raspberry Pi 5 that routes requests between local Gemma and Qwen models while coordinating vision, voice, weather, navigation, memory, and reminder services.

> Implemented real-time object detection using OpenCV, YOLO, and a USB camera with confidence scoring, scene analysis, and annotated previews.

> Built FastAPI-based chat, weather, and navigation dashboards using REST APIs, JavaScript, HTML, and CSS to display AI responses, forecasts, routes, and camera output.

> Integrated Whisper, Piper, SQLite, multithreading, and natural-language parsing to support speech interaction, persistent data, editable reminders, and scheduled alerts.

---

## Author

**Joshua Anicette**

Computer Engineering Student
Interested in artificial intelligence, embedded systems, robotics, software engineering, and entrepreneurship.

---

## License

This project is currently intended for educational, portfolio, and personal-development use.

Add a license file before public distribution. The MIT License is a common choice for open-source software projects:

```text
MIT License
```

---

## Acknowledgments

This project uses or integrates technologies from:

* Raspberry Pi
* Ollama
* Google
* Ultralytics
* OpenCV
* FastAPI
* Whisper
* Piper
* OpenStreetMap
* Leaflet
* Python open-source contributors

---

## Project Status

Jarvis is under active development. Core AI, voice, vision, dashboard, navigation, weather, and reminder capabilities are functional, while robotics control and advanced persistent conversation management remain ongoing development areas.
