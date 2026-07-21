from pathlib import Path
import yaml


class Config:
    def __init__(self, config_path=None):
        root = Path(__file__).resolve().parents[2]
        self.path = Path(config_path) if config_path else root / "config" / "settings.yaml"
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        return yaml.safe_load(self.path.read_text()) or {}

    def get(self, key, default=None):
        return self.data.get(key, default)
