from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UserProfileDatabase:
    """Local structured knowledge store for Jarvis personalization."""

    def __init__(
        self,
        path: str | Path = "data/profile/jarvis_profile.db",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=15.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
        connection.execute(
            "PRAGMA journal_mode = WAL"
        )
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS profile_facts (
            memory_key TEXT PRIMARY KEY,
            note TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'other',
            importance REAL NOT NULL DEFAULT 0.75,
            confidence REAL NOT NULL DEFAULT 1.0,
            source TEXT NOT NULL DEFAULT 'conversation',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS interests (
            topic TEXT PRIMARY KEY,
            category TEXT NOT NULL DEFAULT 'interest',
            score REAL NOT NULL DEFAULT 0.0,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            last_evidence TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS style_metrics (
            name TEXT PRIMARY KEY,
            value REAL NOT NULL,
            samples INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS style_terms (
            term TEXT NOT NULL,
            kind TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (term, kind)
        );

        CREATE TABLE IF NOT EXISTS code_documents (
            path TEXT PRIMARY KEY,
            language TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            line_count INTEGER NOT NULL,
            last_request TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS code_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            request TEXT NOT NULL,
            paths_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS interaction_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            prompt_excerpt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_profile_facts_category
            ON profile_facts(category, active);
        CREATE INDEX IF NOT EXISTS idx_interests_score
            ON interests(score DESC);
        CREATE INDEX IF NOT EXISTS idx_code_documents_updated
            ON code_documents(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_interactions_category
            ON interaction_patterns(category, created_at DESC);
        """
        with self._lock, self._connect() as connection:
            connection.executescript(schema)

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(value).casefold(),
        ).strip()

    @classmethod
    def _fact_key(
        cls,
        note: str,
        category: str,
    ) -> str:
        normalized = (
            cls._normalize(category)
            + ":"
            + cls._normalize(note)
        )
        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()[:24]

    def save_fact(
        self,
        note: str,
        category: str = "other",
        importance: float = 0.75,
        confidence: float = 1.0,
        source: str = "conversation",
    ) -> None:
        cleaned = re.sub(r"\s+", " ", str(note)).strip()
        if not cleaned:
            return
        now = self._now()
        key = self._fact_key(cleaned, category)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO profile_facts (
                    memory_key, note, category, importance,
                    confidence, source, active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(memory_key) DO UPDATE SET
                    note = excluded.note,
                    importance = MAX(profile_facts.importance, excluded.importance),
                    confidence = MAX(profile_facts.confidence, excluded.confidence),
                    source = excluded.source,
                    active = 1,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    cleaned,
                    str(category or "other"),
                    max(0.0, min(1.0, float(importance))),
                    max(0.0, min(1.0, float(confidence))),
                    str(source or "conversation"),
                    now,
                    now,
                ),
            )

    def reinforce_interest(
        self,
        topic: str,
        category: str = "interest",
        strength: float = 0.5,
        evidence: str = "",
    ) -> None:
        cleaned = self._normalize(topic)
        if not cleaned:
            return
        now = self._now()
        amount = max(0.1, min(2.0, float(strength)))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interests (
                    topic, category, score, evidence_count,
                    last_evidence, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    category = excluded.category,
                    score = MIN(20.0, interests.score + excluded.score),
                    evidence_count = interests.evidence_count + 1,
                    last_evidence = excluded.last_evidence,
                    updated_at = excluded.updated_at
                """,
                (
                    cleaned,
                    str(category or "interest"),
                    amount,
                    str(evidence)[:500],
                    now,
                ),
            )

    def record_memory_action(
        self,
        action: dict[str, Any],
        source_text: str = "",
    ) -> None:
        action_name = str(
            action.get("action", "ignore")
        )
        if action_name == "save_confirmed":
            self.save_fact(
                note=action.get("note", ""),
                category=action.get("category", "other"),
                importance=action.get("importance", 0.75),
                confidence=action.get("confidence", 1.0),
                source="memory_analyzer",
            )
        elif action_name == "reinforce_interest":
            self.reinforce_interest(
                topic=action.get("topic", ""),
                category=action.get("category", "interest"),
                strength=action.get("evidence_strength", 0.55),
                evidence=source_text,
            )
        elif action_name == "update_memory" and action.get("note"):
            self.save_fact(
                note=action.get("note", ""),
                category=action.get("category", "other"),
                importance=action.get("importance", 0.75),
                confidence=action.get("confidence", 0.8),
                source="memory_update",
            )
        elif action_name == "forget_memory":
            self.forget_facts(
                str(action.get("query", ""))
            )

    def import_memories(
        self,
        memories: list[dict[str, Any]],
    ) -> int:
        """Copy existing JSON memories into the structured profile database."""
        imported = 0
        for memory in memories:
            if not isinstance(memory, dict):
                continue
            note = str(memory.get("note", "")).strip()
            if not note:
                continue
            self.save_fact(
                note=note,
                category=str(memory.get("category", "other")),
                importance=float(memory.get("importance", 0.75)),
                confidence=float(memory.get("confidence", 0.8)),
                source=str(memory.get("source", "memory_import")),
            )
            imported += 1
        return imported

    def forget_facts(self, query: str) -> int:
        normalized = self._normalize(query)
        if not normalized:
            return 0
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE profile_facts
                SET active = 0, updated_at = ?
                WHERE LOWER(note) LIKE ?
                """,
                (
                    self._now(),
                    f"%{normalized}%",
                ),
            )
            return int(cursor.rowcount)

    def save_style_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> None:
        now = self._now()
        samples = int(snapshot.get("samples", 0))
        signals = snapshot.get("signals", {})
        slang = snapshot.get("slang", {})
        phrases = snapshot.get("phrases", {})
        interests = snapshot.get("interests", {})

        with self._lock, self._connect() as connection:
            for name, value in signals.items():
                connection.execute(
                    """
                    INSERT INTO style_metrics(name, value, samples, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        value = excluded.value,
                        samples = excluded.samples,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(name),
                        float(value),
                        samples,
                        now,
                    ),
                )

            for kind, terms in (
                ("slang", slang),
                ("phrase", phrases),
            ):
                for term, count in terms.items():
                    connection.execute(
                        """
                        INSERT INTO style_terms(term, kind, count, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(term, kind) DO UPDATE SET
                            count = excluded.count,
                            updated_at = excluded.updated_at
                        """,
                        (
                            str(term),
                            kind,
                            int(count),
                            now,
                        ),
                    )

        with self._lock, self._connect() as connection:
            for topic, score in interests.items():
                cleaned = self._normalize(topic)
                if not cleaned:
                    continue
                connection.execute(
                    """
                    INSERT INTO interests (
                        topic, category, score, evidence_count,
                        last_evidence, updated_at
                    )
                    VALUES (?, 'interest', ?, 1, ?, ?)
                    ON CONFLICT(topic) DO UPDATE SET
                        score = MAX(interests.score, excluded.score),
                        last_evidence = excluded.last_evidence,
                        updated_at = excluded.updated_at
                    """,
                    (
                        cleaned,
                        max(0.1, min(20.0, float(score))),
                        "adaptive style profile",
                        now,
                    ),
                )

    @staticmethod
    def _language_for(path: str) -> str:
        suffix = Path(path).suffix.casefold()
        return {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript/React",
            ".jsx": "JavaScript/React",
            ".html": "HTML",
            ".css": "CSS",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".sh": "Shell",
            ".md": "Markdown",
        }.get(suffix, suffix.lstrip(".").upper() or "Text")

    def store_code_document(
        self,
        path: str,
        content: str,
        request: str = "",
    ) -> None:
        text = str(content)
        digest = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO code_documents (
                    path, language, content, content_hash,
                    line_count, last_request, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    language = excluded.language,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    line_count = excluded.line_count,
                    last_request = excluded.last_request,
                    updated_at = excluded.updated_at
                """,
                (
                    str(path),
                    self._language_for(path),
                    text,
                    digest,
                    len(text.splitlines()),
                    str(request)[:1000],
                    self._now(),
                ),
            )

    def get_code_context(
        self,
        query: str,
        max_documents: int = 4,
        max_chars: int = 8_000,
    ) -> str:
        """Retrieve bounded code previously selected by the user."""
        normalized_query = self._normalize(query)
        stop_words = {
            "about", "add", "and", "change", "code", "could", "file",
            "for", "from", "help", "how", "into", "make", "project",
            "should", "that", "the", "this", "update", "want", "with",
        }
        tokens = {
            token
            for token in re.findall(r"[a-z0-9_]+", normalized_query)
            if len(token) >= 3 and token not in stop_words
        }

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT path, language, content, last_request, updated_at
                FROM code_documents
                ORDER BY updated_at DESC
                LIMIT 40
                """
            ).fetchall()

        ranked: list[tuple[int, sqlite3.Row]] = []
        for position, row in enumerate(rows):
            searchable = self._normalize(
                f"{row['path']} {row['last_request'] or ''}"
            )
            score = sum(
                3 if token in self._normalize(row["path"]) else 1
                for token in tokens
                if token in searchable
            )
            if score or (not tokens and position < 2):
                ranked.append((score, row))

        if not ranked:
            return ""

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        sections: list[str] = []
        used_chars = 0
        for _, row in ranked[:max(1, int(max_documents))]:
            header = (
                f"FILE: {row['path']} "
                f"({row['language']}, previously selected)\n---\n"
            )
            remaining = max(0, int(max_chars) - used_chars - len(header) - 5)
            if remaining <= 0:
                break
            content = str(row["content"])[:remaining]
            sections.append(f"{header}{content}\n---")
            used_chars += len(sections[-1])

        return "\n\n".join(sections)

    def record_code_request(
        self,
        mode: str,
        request: str,
        paths: list[str] | set[str],
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO code_requests(mode, request, paths_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(mode),
                    str(request)[:4000],
                    json.dumps(sorted(str(path) for path in paths)),
                    self._now(),
                ),
            )

    def record_interaction(
        self,
        category: str,
        prompt: str,
    ) -> None:
        cleaned = re.sub(r"\s+", " ", str(prompt)).strip()
        if not cleaned:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interaction_patterns(
                    category, prompt_excerpt, created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    str(category or "general"),
                    cleaned[:500],
                    self._now(),
                ),
            )
            connection.execute(
                """
                DELETE FROM interaction_patterns
                WHERE id NOT IN (
                    SELECT id FROM interaction_patterns
                    ORDER BY id DESC LIMIT 1000
                )
                """
            )

    def forget_knowledge(self, query: str) -> int:
        """Remove matching personal knowledge across profile tables."""
        normalized = self._normalize(query)
        if not normalized:
            return 0

        removed = self.forget_facts(normalized)
        like_query = f"%{normalized}%"
        with self._lock, self._connect() as connection:
            for statement, parameters in (
                (
                    "DELETE FROM interests WHERE LOWER(topic) LIKE ?",
                    (like_query,),
                ),
                (
                    """
                    DELETE FROM code_documents
                    WHERE LOWER(path) LIKE ?
                       OR LOWER(COALESCE(last_request, '')) LIKE ?
                    """,
                    (like_query, like_query),
                ),
                (
                    "DELETE FROM code_requests WHERE LOWER(request) LIKE ?",
                    (like_query,),
                ),
                (
                    """
                    DELETE FROM interaction_patterns
                    WHERE LOWER(prompt_excerpt) LIKE ?
                    """,
                    (like_query,),
                ),
            ):
                cursor = connection.execute(statement, parameters)
                removed += int(cursor.rowcount)
        return removed

    def get_profile_summary(self) -> str:
        """Return a readable, privacy-conscious view of saved learning."""
        with self._lock, self._connect() as connection:
            facts = connection.execute(
                """
                SELECT note, category
                FROM profile_facts
                WHERE active = 1
                ORDER BY importance DESC, updated_at DESC
                LIMIT 12
                """
            ).fetchall()
            interests = connection.execute(
                """
                SELECT topic
                FROM interests
                ORDER BY score DESC, updated_at DESC
                LIMIT 10
                """
            ).fetchall()
            terms = connection.execute(
                """
                SELECT term, kind
                FROM style_terms
                ORDER BY count DESC, updated_at DESC
                LIMIT 8
                """
            ).fetchall()
            code_files = connection.execute(
                """
                SELECT path, language
                FROM code_documents
                ORDER BY updated_at DESC
                LIMIT 8
                """
            ).fetchall()

        lines = ["Here is what I currently have in your local profile:"]
        if facts:
            lines.append("Facts and preferences:")
            lines.extend(
                f"- [{row['category']}] {row['note']}"
                for row in facts
            )
        if interests:
            lines.append(
                "Interests: "
                + ", ".join(str(row["topic"]) for row in interests)
                + "."
            )
        if terms:
            lines.append(
                "Tone and phrasing cues: "
                + ", ".join(
                    f"{row['term']} ({row['kind']})"
                    for row in terms
                )
                + "."
            )
        if code_files:
            lines.append("Learned code files:")
            lines.extend(
                f"- {row['path']} ({row['language']})"
                for row in code_files
            )
        if len(lines) == 1:
            lines.append("Nothing has been learned yet.")
        lines.append(
            "Say 'profile forget: <topic>' to remove matching saved knowledge."
        )
        return "\n".join(lines)

    def get_clothing_preferences(
        self,
        limit: int = 5,
    ) -> list[str]:
        terms = (
            "%wear%",
            "%cloth%",
            "%jacket%",
            "%shoe%",
            "%cold%",
            "%hot%",
            "%rain%",
        )
        conditions = " OR ".join(
            "LOWER(note) LIKE ?"
            for _ in terms
        )
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT note
                FROM profile_facts
                WHERE active = 1
                  AND ({conditions})
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (*terms, int(limit)),
            ).fetchall()
        return [
            str(row["note"])
            for row in rows
        ]

    def get_prompt_context(
        self,
        query: str = "",
        fact_limit: int = 8,
        interest_limit: int = 8,
    ) -> str:
        with self._lock, self._connect() as connection:
            facts = connection.execute(
                """
                SELECT note, category, confidence
                FROM profile_facts
                WHERE active = 1
                ORDER BY importance DESC, confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (int(fact_limit),),
            ).fetchall()
            interests = connection.execute(
                """
                SELECT topic, score
                FROM interests
                ORDER BY score DESC, updated_at DESC
                LIMIT ?
                """,
                (int(interest_limit),),
            ).fetchall()
            styles = connection.execute(
                """
                SELECT name, value
                FROM style_metrics
                ORDER BY name
                """
            ).fetchall()
            code_files = connection.execute(
                """
                SELECT path, language, line_count
                FROM code_documents
                ORDER BY updated_at DESC
                LIMIT 6
                """
            ).fetchall()

        if not any((facts, interests, styles, code_files)):
            return ""

        lines = [
            "Local personal profile database context:",
        ]
        if facts:
            lines.append("- Relevant durable facts and preferences:")
            lines.extend(
                f"  - [{row['category']}] {row['note']} "
                f"({float(row['confidence']):.0%} confidence)"
                for row in facts
            )
        if interests:
            lines.append(
                "- Strong interests: "
                + ", ".join(
                    str(row["topic"])
                    for row in interests
                )
                + "."
            )
        if styles:
            lines.append(
                "- Communication signals: "
                + ", ".join(
                    f"{row['name']}={float(row['value']):.0%}"
                    for row in styles
                )
                + "."
            )
        if code_files and any(
            word in self._normalize(query)
            for word in (
                "code", "program", "python", "javascript",
                "engineering", "project", "file",
            )
        ):
            lines.append("- Recently learned code context:")
            lines.extend(
                f"  - {row['path']} ({row['language']}, "
                f"{row['line_count']} lines)"
                for row in code_files
            )

        lines.append(
            "Use this local profile only when relevant. Do not expose stored "
            "details unnecessarily, and treat inferred interests as adjustable."
        )
        return "\n".join(lines)

    def counts(self) -> dict[str, int]:
        tables = (
            "profile_facts",
            "interests",
            "style_metrics",
            "style_terms",
            "code_documents",
            "code_requests",
            "interaction_patterns",
        )
        with self._lock, self._connect() as connection:
            return {
                table: int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                )
                for table in tables
            }
