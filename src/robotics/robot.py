class Robot:
    def __init__(self, motors):
        self.motors = motors

    def move_forward(self, speed=0.5):
        self.motors.forward(speed)

    def move_backward(self, speed=0.5):
        self.motors.backward(speed)

    def turn_left(self, speed=0.5):
        self.motors.left(speed)

    def turn_right(self, speed=0.5):
        self.motors.right(speed)

    def stop(self):
        self.motors.stop()
