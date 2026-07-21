from abc import ABC, abstractmethod


class MotorController(ABC):
    @abstractmethod
    def forward(self, speed: float = 0.5):
        raise NotImplementedError

    @abstractmethod
    def backward(self, speed: float = 0.5):
        raise NotImplementedError

    @abstractmethod
    def left(self, speed: float = 0.5):
        raise NotImplementedError

    @abstractmethod
    def right(self, speed: float = 0.5):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError
