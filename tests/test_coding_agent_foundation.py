from pathlib import Path

import pytest

from src.ai.model_router import ModelRouter
from src.tools.coding_tools import CodingTools


def test_router_uses_coder_for_embedded_work():
    router = ModelRouter()
    decision = router.select_model(
        "debug this UART C++ firmware",
        category="embedded",
    )
    assert decision.model == "qwen2.5-coder:3b"


def test_router_uses_strong_coder_for_driver_work():
    router = ModelRouter()
    decision = router.select_model(
        "debug this Linux device driver linker error",
        category="programming",
    )
    assert decision.model == "qwen2.5-coder:7b"


def test_coding_tools_prevent_path_escape(tmp_path: Path):
    tools = CodingTools(tmp_path)
    with pytest.raises(ValueError):
        tools.read_file("../outside.txt")
