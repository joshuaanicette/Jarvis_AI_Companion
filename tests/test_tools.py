from src.tools.registry import ToolRegistry
from src.tools.time_tool import TimeTool


def test_time_tool():
    registry = ToolRegistry()
    registry.register(TimeTool())
    assert isinstance(registry.execute("time"), str)
