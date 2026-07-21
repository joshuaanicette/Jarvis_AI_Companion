from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ai.memory_policy import (
    MAX_CONFIRMED_MEMORIES,
    MAX_INFERRED_MEMORIES,
    MAX_INTEREST_CANDIDATES,
    MIN_INFERENCE_CONFIDENCE,
    MIN_INFERENCE_EVIDENCE,
    clamp_score,
    normalize_category,
)


class MemoryManager:
    def __init__(
        self,
        memory_path: str | Path = "data/memory/memory.json",
    ):
        self.memory_path = Path(
            memory_path
        )

        self.memory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

        self._data = self._load()

    def _default_data(self) -> dict[str, Any]:
        return {
            "version": 2,
            "profile": {},
            "confirmed_memories": [],
            "inferred_memories": [],
            "interest_candidates": {},
            "user_notes": [],
        }

    def _load(self) -> dict[str, Any]:
        if not self.memory_path.exists():
            data = self._default_data()
            self._write_data(data)
            return data

        try:
            with self.memory_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                loaded = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ):
            loaded = self._default_data()

        return self._migrate_data(
            loaded
        )

    def _migrate_data(
        self,
        loaded: Any,
    ) -> dict[str, Any]:
        default = self._default_data()

        if not isinstance(
            loaded,
            dict,
        ):
            return default

        result = deepcopy(
            default
        )

        result["version"] = 2

        profile = loaded.get(
            "profile",
            {},
        )

        if isinstance(
            profile,
            dict,
        ):
            result["profile"] = profile

        confirmed = loaded.get(
            "confirmed_memories",
            [],
        )

        if isinstance(
            confirmed,
            list,
        ):
            result["confirmed_memories"] = confirmed

        inferred = loaded.get(
            "inferred_memories",
            [],
        )

        if isinstance(
            inferred,
            list,
        ):
            result["inferred_memories"] = inferred

        candidates = loaded.get(
            "interest_candidates",
            {},
        )

        if isinstance(
            candidates,
            dict,
        ):
            result["interest_candidates"] = candidates

        notes = loaded.get(
            "user_notes",
            [],
        )

        if isinstance(
            notes,
            list,
        ):
            result["user_notes"] = notes

        # Preserve old key-value memories.
        reserved_keys = {
            "version",
            "profile",
            "confirmed_memories",
            "inferred_memories",
            "interest_candidates",
            "user_notes",
        }

        for key, value in loaded.items():
            if key in reserved_keys:
                continue

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                result["profile"][key] = value

        # Convert old user notes into confirmed memories.
        existing_notes = {
            self._normalize_text(
                memory.get(
                    "note",
                    "",
                )
            )
            for memory in result[
                "confirmed_memories"
            ]
            if isinstance(
                memory,
                dict,
            )
        }

        for note in result["user_notes"]:
            if not isinstance(
                note,
                str,
            ):
                continue

            normalized = self._normalize_text(
                note
            )

            if (
                not normalized
                or normalized in existing_notes
            ):
                continue

            result[
                "confirmed_memories"
            ].append(
                self._build_memory(
                    note=note,
                    category="other",
                    importance=0.75,
                    confidence=1.0,
                    memory_type="confirmed",
                    source="legacy_user_note",
                )
            )

            existing_notes.add(
                normalized
            )

        self._write_data(
            result
        )

        return result

    def _write_data(
        self,
        data: dict[str, Any],
    ) -> None:
        temporary_path = self.memory_path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(
            self.memory_path
        )

    def save(self) -> None:
        with self._lock:
            self._write_data(
                self._data
            )

    def remember(
        self,
        key: str,
        value: Any,
    ) -> None:
        normalized_key = (
            key.strip()
            .lower()
            .replace(" ", "_")
        )

        if not normalized_key:
            return

        with self._lock:
            self._data[
                "profile"
            ][normalized_key] = value

            self.save()

    def recall(
        self,
        key: str,
    ) -> Any:
        normalized_key = (
            key.strip()
            .lower()
            .replace(" ", "_")
        )

        with self._lock:
            return self._data[
                "profile"
            ].get(
                normalized_key
            )

    def add_note(
        self,
        note: str,
    ) -> dict[str, Any] | None:
        return self.save_confirmed_memory(
            note=note,
            category="other",
            importance=0.75,
            confidence=1.0,
            source="explicit_note",
        )

    def get_notes(self) -> list[str]:
        with self._lock:
            notes = []

            for memory in self._data[
                "confirmed_memories"
            ]:
                note = memory.get(
                    "note"
                )

                if note:
                    notes.append(
                        str(note)
                    )

            return notes

    def save_confirmed_memory(
        self,
        note: str,
        category: str = "other",
        importance: float = 0.75,
        confidence: float = 1.0,
        source: str = "user_statement",
    ) -> dict[str, Any] | None:
        cleaned_note = self._clean_note(
            note
        )

        if not cleaned_note:
            return None

        normalized_note = self._normalize_text(
            cleaned_note
        )

        with self._lock:
            existing = self._find_similar_memory(
                normalized_note,
                self._data[
                    "confirmed_memories"
                ],
            )

            if existing is not None:
                existing[
                    "importance"
                ] = max(
                    clamp_score(
                        existing.get(
                            "importance"
                        ),
                    ),
                    clamp_score(
                        importance,
                        0.75,
                    ),
                )

                existing[
                    "confidence"
                ] = max(
                    clamp_score(
                        existing.get(
                            "confidence"
                        ),
                    ),
                    clamp_score(
                        confidence,
                        1.0,
                    ),
                )

                existing[
                    "updated_at"
                ] = self._now()

                existing[
                    "reinforcement_count"
                ] = int(
                    existing.get(
                        "reinforcement_count",
                        1,
                    )
                ) + 1

                self.save()
                return deepcopy(
                    existing
                )

            memory = self._build_memory(
                note=cleaned_note,
                category=category,
                importance=importance,
                confidence=confidence,
                memory_type="confirmed",
                source=source,
            )

            self._data[
                "confirmed_memories"
            ].append(
                memory
            )

            self._trim_memories()
            self.save()

            return deepcopy(
                memory
            )

    def reinforce_interest(
        self,
        topic: str,
        evidence_strength: float = 0.6,
        category: str = "interest",
        evidence_text: str | None = None,
    ) -> dict[str, Any] | None:
        cleaned_topic = self._clean_topic(
            topic
        )

        if not cleaned_topic:
            return None

        topic_key = self._normalize_text(
            cleaned_topic
        )

        strength = clamp_score(
            evidence_strength,
            0.5,
        )

        with self._lock:
            candidates = self._data[
                "interest_candidates"
            ]

            candidate = candidates.get(
                topic_key
            )

            if candidate is None:
                candidate = {
                    "topic": cleaned_topic,
                    "category": normalize_category(
                        category
                    ),
                    "evidence_count": 0,
                    "total_strength": 0.0,
                    "confidence": 0.0,
                    "examples": [],
                    "created_at": self._now(),
                    "updated_at": self._now(),
                }

                candidates[
                    topic_key
                ] = candidate

            candidate[
                "evidence_count"
            ] = int(
                candidate.get(
                    "evidence_count",
                    0,
                )
            ) + 1

            candidate[
                "total_strength"
            ] = float(
                candidate.get(
                    "total_strength",
                    0.0,
                )
            ) + strength

            count = candidate[
                "evidence_count"
            ]

            average_strength = (
                candidate[
                    "total_strength"
                ]
                / max(
                    1,
                    count,
                )
            )

            repetition_score = min(
                1.0,
                count
                / MIN_INFERENCE_EVIDENCE,
            )

            candidate[
                "confidence"
            ] = round(
                min(
                    1.0,
                    (
                        average_strength
                        * 0.65
                    )
                    + (
                        repetition_score
                        * 0.35
                    ),
                ),
                3,
            )

            candidate[
                "updated_at"
            ] = self._now()

            if evidence_text:
                cleaned_evidence = self._clean_note(
                    evidence_text
                )

                examples = candidate.setdefault(
                    "examples",
                    [],
                )

                if (
                    cleaned_evidence
                    and cleaned_evidence
                    not in examples
                ):
                    examples.append(
                        cleaned_evidence
                    )

                candidate[
                    "examples"
                ] = examples[-5:]

            promoted = None

            if (
                candidate[
                    "evidence_count"
                ]
                >= MIN_INFERENCE_EVIDENCE
                and candidate[
                    "confidence"
                ]
                >= MIN_INFERENCE_CONFIDENCE
            ):
                promoted = self._promote_candidate_locked(
                    topic_key
                )

            self._trim_memories()
            self.save()

            if promoted is not None:
                return {
                    "status": "promoted",
                    "memory": deepcopy(
                        promoted
                    ),
                }

            return {
                "status": "candidate",
                "candidate": deepcopy(
                    candidate
                ),
            }

    def _promote_candidate_locked(
        self,
        topic_key: str,
    ) -> dict[str, Any] | None:
        candidate = self._data[
            "interest_candidates"
        ].get(
            topic_key
        )

        if candidate is None:
            return None

        topic = candidate.get(
            "topic",
            topic_key,
        )

        category = normalize_category(
            candidate.get(
                "category",
                "interest",
            )
        )

        if category == "project":
            note = (
                f"Josh is actively working on or frequently "
                f"asking about {topic}."
            )

        elif category == "skill":
            note = (
                f"Josh is developing skills related to {topic}."
            )

        else:
            note = (
                f"Josh has a recurring interest in {topic}."
            )

        normalized_note = self._normalize_text(
            note
        )

        existing = self._find_similar_memory(
            normalized_note,
            self._data[
                "inferred_memories"
            ],
        )

        if existing is not None:
            existing[
                "confidence"
            ] = max(
                clamp_score(
                    existing.get(
                        "confidence"
                    ),
                ),
                clamp_score(
                    candidate.get(
                        "confidence"
                    ),
                ),
            )

            existing[
                "evidence_count"
            ] = max(
                int(
                    existing.get(
                        "evidence_count",
                        0,
                    )
                ),
                int(
                    candidate.get(
                        "evidence_count",
                        0,
                    )
                ),
            )

            existing[
                "updated_at"
            ] = self._now()

            del self._data[
                "interest_candidates"
            ][topic_key]

            return existing

        memory = self._build_memory(
            note=note,
            category=category,
            importance=0.75,
            confidence=candidate.get(
                "confidence",
                0.70,
            ),
            memory_type="inferred",
            source="question_pattern",
        )

        memory[
            "topic"
        ] = topic

        memory[
            "evidence_count"
        ] = candidate.get(
            "evidence_count",
            0,
        )

        memory[
            "examples"
        ] = candidate.get(
            "examples",
            [],
        )

        self._data[
            "inferred_memories"
        ].append(
            memory
        )

        del self._data[
            "interest_candidates"
        ][topic_key]

        return memory

    def update_memory(
        self,
        memory_id: str,
        note: str | None = None,
        category: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
    ) -> bool:
        with self._lock:
            memory = self._get_memory_by_id_locked(
                memory_id
            )

            if memory is None:
                return False

            if note is not None:
                cleaned = self._clean_note(
                    note
                )

                if cleaned:
                    memory[
                        "note"
                    ] = cleaned

            if category is not None:
                memory[
                    "category"
                ] = normalize_category(
                    category
                )

            if importance is not None:
                memory[
                    "importance"
                ] = clamp_score(
                    importance
                )

            if confidence is not None:
                memory[
                    "confidence"
                ] = clamp_score(
                    confidence
                )

            memory[
                "updated_at"
            ] = self._now()

            self.save()
            return True

    def forget_memory(
        self,
        query: str,
    ) -> int:
        normalized_query = self._normalize_text(
            query
        )

        if not normalized_query:
            return 0

        removed = 0

        with self._lock:
            for key in (
                "confirmed_memories",
                "inferred_memories",
            ):
                original = self._data[
                    key
                ]

                kept = []

                for memory in original:
                    searchable = self._normalize_text(
                        " ".join(
                            [
                                str(
                                    memory.get(
                                        "note",
                                        "",
                                    )
                                ),
                                str(
                                    memory.get(
                                        "topic",
                                        "",
                                    )
                                ),
                                str(
                                    memory.get(
                                        "category",
                                        "",
                                    )
                                ),
                            ]
                        )
                    )

                    if normalized_query in searchable:
                        removed += 1
                    else:
                        kept.append(
                            memory
                        )

                self._data[
                    key
                ] = kept

            candidate_keys = []

            for key, candidate in self._data[
                "interest_candidates"
            ].items():
                searchable = self._normalize_text(
                    " ".join(
                        [
                            key,
                            str(
                                candidate.get(
                                    "topic",
                                    "",
                                )
                            ),
                        ]
                    )
                )

                if normalized_query in searchable:
                    candidate_keys.append(
                        key
                    )

            for key in candidate_keys:
                del self._data[
                    "interest_candidates"
                ][key]

                removed += 1

            if removed:
                self.save()

        return removed

    def get_confirmed_memories(
        self,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(
                self._data[
                    "confirmed_memories"
                ]
            )

    def get_inferred_memories(
        self,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(
                self._data[
                    "inferred_memories"
                ]
            )

    def get_interest_candidates(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return deepcopy(
                self._data[
                    "interest_candidates"
                ]
            )

    def get_all_memories(
        self,
    ) -> list[dict[str, Any]]:
        with self._lock:
            memories = (
                self._data[
                    "confirmed_memories"
                ]
                + self._data[
                    "inferred_memories"
                ]
            )

            return deepcopy(
                memories
            )

    def get_context(
        self,
    ) -> str:
        with self._lock:
            lines = []

            profile = self._data.get(
                "profile",
                {},
            )

            for key, value in profile.items():
                label = key.replace(
                    "_",
                    " ",
                ).title()

                lines.append(
                    f"- {label}: {value}"
                )

            for memory in self._data[
                "confirmed_memories"
            ]:
                note = memory.get(
                    "note"
                )

                if note:
                    lines.append(
                        f"- {note}"
                    )

            for memory in self._data[
                "inferred_memories"
            ]:
                note = memory.get(
                    "note"
                )

                confidence = memory.get(
                    "confidence",
                    0.0,
                )

                if note:
                    lines.append(
                        f"- Possible inference "
                        f"({confidence:.0%} confidence): "
                        f"{note}"
                    )

            return "\n".join(
                lines
            )

    def record_memory_use(
        self,
        memory_ids: list[str],
    ) -> None:
        if not memory_ids:
            return

        requested = set(
            memory_ids
        )

        with self._lock:
            changed = False

            for memory in self.get_all_memories():
                memory_id = memory.get(
                    "id"
                )

                if memory_id not in requested:
                    continue

                stored = self._get_memory_by_id_locked(
                    memory_id
                )

                if stored is None:
                    continue

                stored[
                    "use_count"
                ] = int(
                    stored.get(
                        "use_count",
                        0,
                    )
                ) + 1

                stored[
                    "last_used_at"
                ] = self._now()

                changed = True

            if changed:
                self.save()

    def _get_memory_by_id_locked(
        self,
        memory_id: str,
    ) -> dict[str, Any] | None:
        for key in (
            "confirmed_memories",
            "inferred_memories",
        ):
            for memory in self._data[
                key
            ]:
                if memory.get(
                    "id"
                ) == memory_id:
                    return memory

        return None

    def _find_similar_memory(
        self,
        normalized_note: str,
        memories: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        target_tokens = set(
            normalized_note.split()
        )

        for memory in memories:
            existing_note = self._normalize_text(
                memory.get(
                    "note",
                    "",
                )
            )

            if not existing_note:
                continue

            if (
                normalized_note == existing_note
                or normalized_note in existing_note
                or existing_note in normalized_note
            ):
                return memory

            existing_tokens = set(
                existing_note.split()
            )

            union = (
                target_tokens
                | existing_tokens
            )

            if not union:
                continue

            overlap = (
                target_tokens
                & existing_tokens
            )

            similarity = (
                len(overlap)
                / len(union)
            )

            if similarity >= 0.72:
                return memory

        return None

    def _trim_memories(
        self,
    ) -> None:
        self._data[
            "confirmed_memories"
        ] = self._data[
            "confirmed_memories"
        ][-MAX_CONFIRMED_MEMORIES:]

        self._data[
            "inferred_memories"
        ] = self._data[
            "inferred_memories"
        ][-MAX_INFERRED_MEMORIES:]

        candidates = self._data[
            "interest_candidates"
        ]

        if len(
            candidates
        ) <= MAX_INTEREST_CANDIDATES:
            return

        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: (
                item[1].get(
                    "evidence_count",
                    0,
                ),
                item[1].get(
                    "confidence",
                    0.0,
                ),
            ),
            reverse=True,
        )

        self._data[
            "interest_candidates"
        ] = dict(
            sorted_candidates[
                :MAX_INTEREST_CANDIDATES
            ]
        )

    def _build_memory(
        self,
        note: str,
        category: str,
        importance: float,
        confidence: float,
        memory_type: str,
        source: str,
    ) -> dict[str, Any]:
        timestamp = self._now()

        safe_category = normalize_category(
            category
        )

        memory_id = (
            f"{memory_type}_"
            f"{int(datetime.now().timestamp() * 1000000)}"
        )

        return {
            "id": memory_id,
            "note": self._clean_note(
                note
            ),
            "category": safe_category,
            "importance": clamp_score(
                importance,
                0.75,
            ),
            "confidence": clamp_score(
                confidence,
                1.0,
            ),
            "memory_type": memory_type,
            "source": source,
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_used_at": None,
            "use_count": 0,
            "reinforcement_count": 1,
        }

    @staticmethod
    def _clean_note(
        note: Any,
    ) -> str:
        if note is None:
            return ""

        cleaned = re.sub(
            r"\s+",
            " ",
            str(note),
        ).strip()

        return cleaned[:500]

    @staticmethod
    def _clean_topic(
        topic: Any,
    ) -> str:
        if topic is None:
            return ""

        cleaned = re.sub(
            r"\s+",
            " ",
            str(topic),
        ).strip(
            " .,!?:;"
        )

        return cleaned[:100]

    @staticmethod
    def _normalize_text(
        text: Any,
    ) -> str:
        normalized = str(
            text or ""
        ).lower()

        normalized = re.sub(
            r"[^a-z0-9\s]",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()