from __future__ import annotations

import re
from typing import Any

from src.ai.memory_policy import (
    MAX_RETRIEVED_MEMORIES,
)


class MemoryRetriever:
    def __init__(
        self,
        memory_manager,
        max_results: int = MAX_RETRIEVED_MEMORIES,
    ):
        self.memory_manager = memory_manager
        self.max_results = max_results

    def retrieve(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        result_limit = (
            limit
            if limit is not None
            else self.max_results
        )

        memories = (
            self.memory_manager.get_all_memories()
        )

        query_tokens = self._tokens(
            query
        )

        scored = []

        for memory in memories:
            score = self._score_memory(
                memory=memory,
                query_tokens=query_tokens,
                query=query,
            )

            if score <= 0.0:
                continue

            scored.append(
                (
                    score,
                    memory,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = [
            memory
            for _, memory in scored[
                :result_limit
            ]
        ]

        memory_ids = [
            memory.get(
                "id"
            )
            for memory in selected
            if memory.get(
                "id"
            )
        ]

        self.memory_manager.record_memory_use(
            memory_ids
        )

        return selected

    def get_context(
        self,
        query: str,
        limit: int | None = None,
    ) -> str:
        memories = self.retrieve(
            query=query,
            limit=limit,
        )

        if not memories:
            return ""

        lines = [
            "Relevant information about the user:",
        ]

        for memory in memories:
            note = memory.get(
                "note",
                "",
            )

            memory_type = memory.get(
                "memory_type",
                "confirmed",
            )

            confidence = float(
                memory.get(
                    "confidence",
                    1.0,
                )
            )

            if not note:
                continue

            if memory_type == "inferred":
                lines.append(
                    f"- Possible inference "
                    f"({confidence:.0%} confidence): "
                    f"{note}"
                )
            else:
                lines.append(
                    f"- {note}"
                )

        lines.append(
            "Use confirmed memories normally. "
            "Treat inferred memories as tentative and "
            "do not present them as unquestionable facts."
        )

        return "\n".join(
            lines
        )

    def _score_memory(
        self,
        memory: dict[str, Any],
        query_tokens: set[str],
        query: str,
    ) -> float:
        note = str(
            memory.get(
                "note",
                "",
            )
        )

        category = str(
            memory.get(
                "category",
                "",
            )
        )

        topic = str(
            memory.get(
                "topic",
                "",
            )
        )

        searchable = " ".join(
            [
                note,
                category,
                topic,
            ]
        )

        memory_tokens = self._tokens(
            searchable
        )

        importance = float(
            memory.get(
                "importance",
                0.5,
            )
        )

        confidence = float(
            memory.get(
                "confidence",
                0.5,
            )
        )

        use_count = int(
            memory.get(
                "use_count",
                0,
            )
        )

        overlap = len(
            query_tokens
            & memory_tokens
        )

        overlap_score = 0.0

        if query_tokens:
            overlap_score = (
                overlap
                / len(
                    query_tokens
                )
            )

        note_normalized = self._normalize(
            note
        )

        query_normalized = self._normalize(
            query
        )

        phrase_bonus = 0.0

        if (
            note_normalized
            and query_normalized
            and (
                note_normalized
                in query_normalized
                or query_normalized
                in note_normalized
            )
        ):
            phrase_bonus = 0.4

        universal_preference_bonus = 0.0

        if category in {
            "preference",
            "communication",
            "identity",
        }:
            universal_preference_bonus = 0.20

        project_bonus = 0.0

        if (
            category in {
                "project",
                "hardware",
                "software",
                "skill",
            }
            and overlap > 0
        ):
            project_bonus = 0.25

        usage_bonus = min(
            0.15,
            use_count * 0.01,
        )

        score = (
            overlap_score
            + phrase_bonus
            + universal_preference_bonus
            + project_bonus
            + usage_bonus
            + (
                importance
                * 0.20
            )
            + (
                confidence
                * 0.10
            )
        )

        # Avoid injecting unrelated inferred memories.
        if (
            memory.get(
                "memory_type"
            ) == "inferred"
            and overlap == 0
        ):
            score -= 0.45

        return score

    @staticmethod
    def _tokens(
        text: str,
    ) -> set[str]:
        normalized = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text.lower(),
        )

        words = {
            word
            for word in normalized.split()
            if len(
                word
            ) >= 3
        }

        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "what",
            "how",
            "why",
            "from",
            "into",
            "about",
            "user",
            "josh",
        }

        return words - stop_words

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        normalized = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text.lower(),
        )

        return re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()