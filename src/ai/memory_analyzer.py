from __future__ import annotations

import json
import re
from typing import Any

from src.ai.memory_policy import (
    EXPLICIT_MEMORY_PREFIXES,
    FORGET_PREFIXES,
    MIN_CONFIRMED_IMPORTANCE,
    NON_MEMORY_PHRASES,
    SENSITIVE_CATEGORIES,
    TEMPORARY_REQUEST_TERMS,
    VALID_MEMORY_ACTIONS,
    clamp_score,
    normalize_category,
)
from src.core.logger import logger


class MemoryAnalyzer:
    def __init__(
        self,
        llm,
        model: str = "gemma3:1b",
    ):
        self.llm = llm
        self.model = model

    def analyze(
        self,
        user_text: str,
        existing_memories: list[dict[str, Any]] | None = None,
        interest_candidates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cleaned_text = self._clean_text(
            user_text
        )

        if not cleaned_text:
            return {
                "action": "ignore",
                "reason": "The message was empty.",
            }

        explicit_action = (
            self._check_explicit_memory_command(
                cleaned_text
            )
        )

        if explicit_action is not None:
            return explicit_action

        if self._is_obviously_temporary(
            cleaned_text
        ):
            return {
                "action": "ignore",
                "reason": (
                    "The message appears temporary "
                    "or transactional."
                ),
            }

        prompt = self._build_prompt(
            user_text=cleaned_text,
            existing_memories=(
                existing_memories
                or []
            ),
            interest_candidates=(
                interest_candidates
                or {}
            ),
        )

        try:
            raw_result = self._generate(
                prompt
            )

            parsed = self._parse_json(
                raw_result
            )

            return self._validate_action(
                parsed
            )

        except Exception as error:
            logger.warning(
                "Memory analysis failed: %s",
                error,
            )

            return self._fallback_analysis(
                cleaned_text
            )

    def _generate(
        self,
        prompt: str,
    ) -> str:
        generate = getattr(
            self.llm,
            "generate",
            None,
        )

        if callable(
            generate
        ):
            try:
                result = generate(
                    prompt=prompt,
                    model=self.model,
                )
            except TypeError:
                try:
                    result = generate(
                        prompt,
                        model=self.model,
                    )
                except TypeError:
                    result = generate(
                        prompt
                    )

            return self._extract_text(
                result
            )

        chat = getattr(
            self.llm,
            "chat",
            None,
        )

        if callable(
            chat
        ):
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            try:
                result = chat(
                    messages=messages,
                    model=self.model,
                )
            except TypeError:
                try:
                    result = chat(
                        messages,
                        model=self.model,
                    )
                except TypeError:
                    result = chat(
                        messages
                    )

            return self._extract_text(
                result
            )

        raise AttributeError(
            "The configured LLM does not provide "
            "generate() or chat()."
        )

    def _build_prompt(
        self,
        user_text: str,
        existing_memories: list[dict[str, Any]],
        interest_candidates: dict[str, Any],
    ) -> str:
        memory_summary = []

        for memory in existing_memories[:12]:
            note = memory.get(
                "note",
                "",
            )

            category = memory.get(
                "category",
                "other",
            )

            if note:
                memory_summary.append(
                    f"- [{category}] {note}"
                )

        candidate_summary = []

        for candidate in list(
            interest_candidates.values()
        )[:12]:
            topic = candidate.get(
                "topic",
                "",
            )

            count = candidate.get(
                "evidence_count",
                0,
            )

            confidence = candidate.get(
                "confidence",
                0.0,
            )

            if topic:
                candidate_summary.append(
                    f"- {topic}: evidence={count}, "
                    f"confidence={confidence}"
                )

        memories_text = (
            "\n".join(
                memory_summary
            )
            or "None"
        )

        candidates_text = (
            "\n".join(
                candidate_summary
            )
            or "None"
        )

        return f"""
You are the private memory evaluator for Joe, a local AI assistant.

Analyze only the user's latest message.

Your goal is to help Joe gradually learn about the user from:
- direct personal statements
- stable preferences
- recurring technical interests
- active projects
- long-term goals
- hardware and software choices
- communication and learning preferences
- repeated types of questions

Do not save:
- current weather or time requests
- greetings
- one-time commands
- accidental speech
- temporary errors
- information found only in Joe's response
- unsupported assumptions
- highly sensitive information unless the user explicitly asked to remember it

Use save_confirmed when the user directly states something durable.

Use reinforce_interest when the question provides evidence of a recurring
interest, project, skill area, or learning goal but does not directly state
a permanent personal fact.

Use update_memory only when the latest message clearly corrects or expands
an existing memory.

Use forget_memory when the user explicitly requests that something be forgotten.

Use ignore when no useful durable information is present.

Important distinction:
A question such as "How do I run Qwen on Raspberry Pi?" may reinforce
"local AI" or "Raspberry Pi development," but it does not prove that the
user is an expert.

Existing memories:
{memories_text}

Existing interest candidates:
{candidates_text}

Return only one JSON object.

Allowed formats:

{{
  "action": "ignore",
  "reason": "brief reason"
}}

{{
  "action": "save_confirmed",
  "note": "short durable memory written in third person",
  "category": "identity|preference|interest|project|goal|hardware|software|decision|routine|skill|education|career|communication|other",
  "importance": 0.0,
  "confidence": 0.0,
  "reason": "brief reason"
}}

{{
  "action": "reinforce_interest",
  "topic": "short topic",
  "category": "interest|project|skill|education|career|software|hardware",
  "evidence_strength": 0.0,
  "reason": "brief reason"
}}

{{
  "action": "update_memory",
  "memory_id": "existing memory id",
  "note": "updated memory",
  "category": "valid category",
  "importance": 0.0,
  "confidence": 0.0,
  "reason": "brief reason"
}}

{{
  "action": "forget_memory",
  "query": "what should be forgotten",
  "reason": "brief reason"
}}

User message:
{user_text}
""".strip()

    def _validate_action(
        self,
        action: Any,
    ) -> dict[str, Any]:
        if not isinstance(
            action,
            dict,
        ):
            return {
                "action": "ignore",
                "reason": (
                    "The memory evaluator returned "
                    "an invalid response."
                ),
            }

        action_name = str(
            action.get(
                "action",
                "ignore",
            )
        ).strip().lower()

        if action_name not in VALID_MEMORY_ACTIONS:
            action_name = "ignore"

        if action_name == "save_confirmed":
            note = self._clean_text(
                action.get(
                    "note",
                    "",
                )
            )

            importance = clamp_score(
                action.get(
                    "importance"
                ),
                0.70,
            )

            confidence = clamp_score(
                action.get(
                    "confidence"
                ),
                0.90,
            )

            category = normalize_category(
                action.get(
                    "category"
                )
            )

            if (
                not note
                or importance
                < MIN_CONFIRMED_IMPORTANCE
            ):
                return {
                    "action": "ignore",
                    "reason": (
                        "The proposed confirmed memory "
                        "was not important enough."
                    ),
                }

            if (
                category in SENSITIVE_CATEGORIES
                and not action.get(
                    "explicit_request",
                    False,
                )
            ):
                return {
                    "action": "ignore",
                    "reason": (
                        "Sensitive information was not "
                        "explicitly requested for memory."
                    ),
                }

            return {
                "action": "save_confirmed",
                "note": note,
                "category": category,
                "importance": importance,
                "confidence": confidence,
                "reason": self._clean_text(
                    action.get(
                        "reason",
                        "",
                    )
                ),
            }

        if action_name == "reinforce_interest":
            topic = self._clean_topic(
                action.get(
                    "topic",
                    "",
                )
            )

            if not topic:
                return {
                    "action": "ignore",
                    "reason": (
                        "No valid interest topic "
                        "was returned."
                    ),
                }

            return {
                "action": "reinforce_interest",
                "topic": topic,
                "category": normalize_category(
                    action.get(
                        "category",
                        "interest",
                    )
                ),
                "evidence_strength": clamp_score(
                    action.get(
                        "evidence_strength"
                    ),
                    0.55,
                ),
                "reason": self._clean_text(
                    action.get(
                        "reason",
                        "",
                    )
                ),
            }

        if action_name == "update_memory":
            memory_id = self._clean_text(
                action.get(
                    "memory_id",
                    "",
                )
            )

            note = self._clean_text(
                action.get(
                    "note",
                    "",
                )
            )

            if not memory_id:
                return {
                    "action": "ignore",
                    "reason": (
                        "No memory ID was supplied "
                        "for the update."
                    ),
                }

            return {
                "action": "update_memory",
                "memory_id": memory_id,
                "note": note or None,
                "category": normalize_category(
                    action.get(
                        "category"
                    )
                ),
                "importance": clamp_score(
                    action.get(
                        "importance"
                    ),
                    0.75,
                ),
                "confidence": clamp_score(
                    action.get(
                        "confidence"
                    ),
                    0.90,
                ),
                "reason": self._clean_text(
                    action.get(
                        "reason",
                        "",
                    )
                ),
            }

        if action_name == "forget_memory":
            query = self._clean_text(
                action.get(
                    "query",
                    "",
                )
            )

            if not query:
                return {
                    "action": "ignore",
                    "reason": (
                        "No memory query was supplied "
                        "for deletion."
                    ),
                }

            return {
                "action": "forget_memory",
                "query": query,
                "reason": self._clean_text(
                    action.get(
                        "reason",
                        "",
                    )
                ),
            }

        return {
            "action": "ignore",
            "reason": self._clean_text(
                action.get(
                    "reason",
                    "No durable information was found.",
                )
            ),
        }

    def _check_explicit_memory_command(
        self,
        text: str,
    ) -> dict[str, Any] | None:
        normalized = text.lower().strip()

        for prefix in FORGET_PREFIXES:
            if not normalized.startswith(
                prefix
            ):
                continue

            query = text[
                len(prefix):
            ].strip(
                " :,.!?"
            )

            if not query:
                return {
                    "action": "ignore",
                    "reason": (
                        "The user did not specify "
                        "what to forget."
                    ),
                }

            return {
                "action": "forget_memory",
                "query": query,
                "reason": (
                    "The user explicitly requested "
                    "that this be forgotten."
                ),
            }

        for prefix in EXPLICIT_MEMORY_PREFIXES:
            if not normalized.startswith(
                prefix
            ):
                continue

            note = text[
                len(prefix):
            ].strip(
                " :,.!?"
            )

            if not note:
                return {
                    "action": "ignore",
                    "reason": (
                        "The user did not provide "
                        "a memory to save."
                    ),
                }

            return {
                "action": "save_confirmed",
                "note": note,
                "category": "other",
                "importance": 1.0,
                "confidence": 1.0,
                "explicit_request": True,
                "reason": (
                    "The user explicitly requested "
                    "that this be remembered."
                ),
            }

        return None

    def _fallback_analysis(
        self,
        text: str,
    ) -> dict[str, Any]:
        normalized = text.lower()

        interest_rules = {
            "raspberry pi": (
                "Raspberry Pi development",
                "hardware",
            ),
            "qwen": (
                "local language models",
                "software",
            ),
            "ollama": (
                "local language models",
                "software",
            ),
            "computer vision": (
                "computer vision",
                "interest",
            ),
            "opencv": (
                "computer vision",
                "software",
            ),
            "yolo": (
                "object detection",
                "software",
            ),
            "camera": (
                "computer vision and camera systems",
                "interest",
            ),
            "robot": (
                "robotics",
                "interest",
            ),
            "rover": (
                "robotics",
                "project",
            ),
            "engineering": (
                "engineering",
                "education",
            ),
            "python": (
                "Python programming",
                "skill",
            ),
            "linux": (
                "Linux systems",
                "skill",
            ),
        }

        for keyword, (
            topic,
            category,
        ) in interest_rules.items():
            if keyword in normalized:
                return {
                    "action": "reinforce_interest",
                    "topic": topic,
                    "category": category,
                    "evidence_strength": 0.55,
                    "reason": (
                        "Rule-based fallback detected "
                        "a potentially recurring topic."
                    ),
                }

        preference_patterns = (
            (
                r"\bi prefer\s+(.+)",
                "preference",
            ),
            (
                r"\bi like\s+(.+)",
                "interest",
            ),
            (
                r"\bi am building\s+(.+)",
                "project",
            ),
            (
                r"\bi'm building\s+(.+)",
                "project",
            ),
            (
                r"\bi want to learn\s+(.+)",
                "goal",
            ),
            (
                r"\bmy goal is\s+(.+)",
                "goal",
            ),
        )

        for pattern, category in preference_patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            value = match.group(
                1
            ).strip(
                " .,!?"
            )

            if not value:
                continue

            return {
                "action": "save_confirmed",
                "note": (
                    f"Josh stated that {value}."
                ),
                "category": category,
                "importance": 0.75,
                "confidence": 0.85,
                "reason": (
                    "Rule-based fallback detected "
                    "a direct personal statement."
                ),
            }

        return {
            "action": "ignore",
            "reason": (
                "No durable information was detected "
                "by the fallback analyzer."
            ),
        }

    def _is_obviously_temporary(
        self,
        text: str,
    ) -> bool:
        normalized = text.lower().strip()

        if normalized in NON_MEMORY_PHRASES:
            return True

        if any(
            term in normalized
            for term in TEMPORARY_REQUEST_TERMS
        ):
            personal_markers = (
                "i prefer",
                "i like",
                "i am",
                "i'm",
                "my goal",
                "remember",
            )

            if not any(
                marker in normalized
                for marker in personal_markers
            ):
                return True

        return False

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict[str, Any]:
        cleaned = text.strip()

        cleaned = re.sub(
            r"^```(?:json)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"```$",
            "",
            cleaned,
        ).strip()

        try:
            parsed = json.loads(
                cleaned
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if not match:
            raise ValueError(
                "The memory model did not return JSON."
            )

        parsed = json.loads(
            match.group(0)
        )

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "The memory model returned a non-object JSON value."
            )

        return parsed

    @staticmethod
    def _extract_text(
        result: Any,
    ) -> str:
        if isinstance(
            result,
            str,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):
            for key in (
                "response",
                "content",
                "text",
                "message",
            ):
                value = result.get(
                    key
                )

                if isinstance(
                    value,
                    str,
                ):
                    return value

                if (
                    key == "message"
                    and isinstance(
                        value,
                        dict,
                    )
                ):
                    content = value.get(
                        "content"
                    )

                    if isinstance(
                        content,
                        str,
                    ):
                        return content

        return str(
            result
        )

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value),
        ).strip()[:500]

    @staticmethod
    def _clean_topic(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value),
        ).strip(
            " .,!?:;"
        )[:100]