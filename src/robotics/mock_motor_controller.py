from src.robotics.motor_controller import MotorController
from src.core.logger import logger


class MockMotorController(MotorController):
    def forward(self, speed=0.5):
        logger.info("Mock motors: forward at %.2f", speed)

    def backward(self, speed=0.5):
        logger.info("Mock motors: backward at %.2f", speed)

    def left(self, speed=0.5):
        logger.info("Mock motors: left at %.2f", speed)

    def right(self, speed=0.5):
        logger.info("Mock motors: right at %.2f", speed)

    def stop(self):
        logger.info("Mock motors: stop")
