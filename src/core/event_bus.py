from collections import defaultdict
from collections.abc import Callable
from src.core.events import Event


class EventBus:
    def __init__(self):
        self._listeners = defaultdict(list)

    def subscribe(self, name: str, callback: Callable[[Event], None]) -> None:
        self._listeners[name].append(callback)

    def publish(self, event: Event) -> None:
        for callback in list(self._listeners.get(event.name, [])):
            callback(event)
