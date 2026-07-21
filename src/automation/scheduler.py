import threading
import time


class Scheduler:
    def __init__(self):
        self.tasks = []
        self.running = False

    def every(self, seconds, callback):
        self.tasks.append({"interval": seconds, "last": 0.0, "callback": callback})

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            now = time.time()
            for task in self.tasks:
                if now - task["last"] >= task["interval"]:
                    task["callback"]()
                    task["last"] = now
            time.sleep(0.1)

    def stop(self):
        self.running = False
