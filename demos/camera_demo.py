import cv2
from src.vision.camera import Camera


def main():
    camera = Camera()
    camera.start()
    try:
        while True:
            frame = camera.read()
            cv2.imshow("Jay Camera", frame.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
