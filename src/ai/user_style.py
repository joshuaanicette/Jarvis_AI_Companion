from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any


class UserStyleManager:
    """Learns durable communication preferences without copying every message."""

    SLANG = {
        "bro", "bruh", "bet", "fr", "ngl", "lowkey", "highkey", "ight",
        "nah", "yo", "lmao", "lol", "w", "goat", "valid", "chill",
    }
    HUMOR_MARKERS = {"lol", "lmao", "😂", "😭", "joke", "funny"}
    FORMAL_MARKERS = {"please", "therefore", "however", "could you", "would you"}

    def __init__(
        self,
        path: str | Path = "data/memory/user_style.json",
        profile_database=None,
    ) -> None:
        self.path = Path(path)
        self.profile_database = profile_database
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data = self._load()
        self._sync_database()

    @staticmethod
    def _default() -> dict[str, Any]:
        return {
            "version": 1,
            "samples": 0,
            "signals": {
                "concise": 0.5,
                "casual": 0.5,
                "humor": 0.25,
                "step_by_step": 0.5,
            },
            "slang": {},
            "phrases": {},
            "interests": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else self._default()
        except (OSError, json.JSONDecodeError):
            return self._default()

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _ema(old: float, observation: float, weight: float = 0.12) -> float:
        return round(max(0.0, min(1.0, old * (1.0 - weight) + observation * weight)), 3)

    def observe(self, text: str) -> None:
        cleaned = str(text).strip()
        if len(cleaned) < 3:
            return
        lower = cleaned.casefold()
        words = re.findall(r"[a-z0-9']+", lower)
        if not words:
            return

        with self._lock:
            signals = self._data["signals"]
            signals["concise"] = self._ema(
                float(signals.get("concise", 0.5)),
                1.0 if len(words) <= 25 else 0.0,
            )
            casual_hits = sum(word in self.SLANG for word in words)
            formal_hits = sum(marker in lower for marker in self.FORMAL_MARKERS)
            signals["casual"] = self._ema(
                float(signals.get("casual", 0.5)),
                1.0 if casual_hits or (not formal_hits and len(words) < 35) else 0.0,
            )
            signals["humor"] = self._ema(
                float(signals.get("humor", 0.25)),
                1.0 if any(marker in lower for marker in self.HUMOR_MARKERS) else 0.0,
            )
            signals["step_by_step"] = self._ema(
                float(signals.get("step_by_step", 0.5)),
                1.0 if any(p in lower for p in ("step by step", "show work", "full detail", "explain")) else 0.0,
            )

            slang_counts = Counter(word for word in words if word in self.SLANG)
            for word, count in slang_counts.items():
                self._data["slang"][word] = int(self._data["slang"].get(word, 0)) + count

            for phrase in ("step by step", "full detail", "make it short", "give me the full file"):
                if phrase in lower:
                    self._data["phrases"][phrase] = int(self._data["phrases"].get(phrase, 0)) + 1

            self._data["samples"] = int(self._data.get("samples", 0)) + 1
            self._trim()
            self._save()
            self._sync_database()

    def reinforce_interest(self, topic: str, strength: float = 1.0) -> None:
        key = re.sub(r"\s+", " ", str(topic).casefold()).strip()
        if not key:
            return
        with self._lock:
            current = float(self._data["interests"].get(key, 0.0))
            self._data["interests"][key] = round(min(20.0, current + max(0.1, strength)), 2)
            self._trim()
            self._save()
            self._sync_database()

    def _trim(self) -> None:
        for key, limit in (("slang", 30), ("phrases", 20), ("interests", 40)):
            items = sorted(self._data[key].items(), key=lambda item: item[1], reverse=True)[:limit]
            self._data[key] = dict(items)

    def get_prompt_context(self) -> str:
        with self._lock:
            signals = dict(self._data.get("signals", {}))
            slang = [word for word, count in self._data.get("slang", {}).items() if count >= 2][:8]
            phrases = [phrase for phrase, count in self._data.get("phrases", {}).items() if count >= 2][:6]
            interests = [
                topic for topic, score in sorted(
                    self._data.get("interests", {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:8]
            ]

        instructions = [
            "Adaptive user style:",
            f"- Concision preference: {signals.get('concise', 0.5):.0%}.",
            f"- Casual tone preference: {signals.get('casual', 0.5):.0%}.",
            f"- Humor preference: {signals.get('humor', 0.25):.0%}.",
            f"- Step-by-step preference: {signals.get('step_by_step', 0.5):.0%}.",
        ]
        if slang:
            instructions.append("- Familiar slang: " + ", ".join(slang) + ". Use sparingly and naturally.")
        if phrases:
            instructions.append("- Repeated phrasing preferences: " + ", ".join(phrases) + ".")
        if interests:
            instructions.append("- Recurring interests: " + ", ".join(interests) + ".")
        instructions.append(
            "- Match the user's energy, clarity, and humor level; do not impersonate them, overuse slang, "
            "repeat typos, or sacrifice technical accuracy."
        )
        return "\n".join(instructions)

    def _sync_database(self) -> None:
        if self.profile_database is None:
            return
        try:
            self.profile_database.save_style_snapshot(
                self.snapshot()
            )
        except Exception:
            # Personalization should never prevent Jarvis from responding.
            return

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))
