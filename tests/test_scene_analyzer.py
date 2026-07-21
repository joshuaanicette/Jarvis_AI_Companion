from src.vision.frame import VisionFrame
from src.vision.scene_analyzer import SceneAnalyzer


def test_scene_description():
    frame = VisionFrame(image=None)
    frame.objects = [{"label": "person"}, {"label": "chair"}]
    result = SceneAnalyzer().describe(frame)
    assert "person" in result
    assert "chair" in result
