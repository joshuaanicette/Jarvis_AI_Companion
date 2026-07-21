from src.vision.camera import Camera
from src.vision.detector import ObjectDetector
from src.vision.scene_analyzer import SceneAnalyzer
from src.vision.vision_manager import VisionManager


def main():
    manager = VisionManager(Camera(), ObjectDetector(), SceneAnalyzer())
    manager.start()
    try:
        frame = manager.process_once()
        print(frame.scene)
        print(frame.objects)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
