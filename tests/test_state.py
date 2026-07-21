from src.core.state import JayState


def test_state_update():
    state = JayState()
    state.update("online", True)
    assert state.online is True
