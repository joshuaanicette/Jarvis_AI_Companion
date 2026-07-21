from src.core.event_bus import EventBus
from src.core.events import Event


def test_publish_subscribe():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda event: received.append(event.data))
    bus.publish(Event("test", 123))
    assert received == [123]
