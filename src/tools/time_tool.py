from datetime import datetime
from src.tools.tool import Tool


class TimeTool(Tool):
    @property
    def name(self):
        return "time"

    def execute(self):
        return datetime.now().strftime("%I:%M %p")
