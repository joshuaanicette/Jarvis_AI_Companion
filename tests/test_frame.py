import numpy as np
from src.vision.frame import VisionFrame


def test_frame_defaults():
    frame = VisionFrame(np.zeros((10, 10, 3)))
    assert frame.objects == []
    assert frame.scene == ""
