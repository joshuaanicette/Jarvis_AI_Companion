from src.ai.memory import MemoryManager


def test_memory_add_and_reload(tmp_path):
    path = tmp_path / "memory.json"
    memory = MemoryManager(path)
    memory.add_memory("preferences", "Likes Marvel", 0.95)

    loaded = MemoryManager(path)
    assert loaded.get_all()["preferences"][0]["value"] == "Likes Marvel"
