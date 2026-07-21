# Jay Architecture

Core application owns shared services.

Conversation flow:

User -> Memory Analyzer -> Memory Store -> Tool Router -> Memory Retriever
-> Ollama -> TTS

Vision flow:

Camera -> VisionFrame -> ObjectDetector -> SceneAnalyzer -> VisionManager

Robotics flow:

Tool/Event -> Robot -> MotorController -> Hardware driver
