from __future__ import annotations


VALID_MEMORY_CATEGORIES = {
    "identity",
    "preference",
    "interest",
    "project",
    "goal",
    "hardware",
    "software",
    "decision",
    "routine",
    "skill",
    "education",
    "career",
    "communication",
    "other",
}


VALID_MEMORY_ACTIONS = {
    "ignore",
    "save_confirmed",
    "reinforce_interest",
    "update_memory",
    "forget_memory",
}


MIN_CONFIRMED_IMPORTANCE = 0.60

MIN_INFERENCE_EVIDENCE = 3

MIN_INFERENCE_CONFIDENCE = 0.70

MAX_RETRIEVED_MEMORIES = 8

MAX_CONFIRMED_MEMORIES = 200

MAX_INFERRED_MEMORIES = 100

MAX_INTEREST_CANDIDATES = 100


TEMPORARY_REQUEST_TERMS = {
    "weather",
    "forecast",
    "temperature",
    "current time",
    "what time",
    "today's weather",
    "tomorrow's weather",
}


NON_MEMORY_PHRASES = {
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
    "what can you do",
    "who are you",
}


SENSITIVE_CATEGORIES = {
    "medical",
    "health",
    "religion",
    "politics",
    "sexuality",
    "financial_account",
    "password",
    "authentication",
    "precise_location",
}


EXPLICIT_MEMORY_PREFIXES = (
    "remember that",
    "remember",
    "save this",
    "save that",
    "add this to memory",
    "keep in mind that",
    "note that",
)


FORGET_PREFIXES = (
    "forget that",
    "forget",
    "remove that memory",
    "delete that memory",
)


def normalize_category(
    category: str | None,
) -> str:
    if not category:
        return "other"

    normalized = (
        str(category)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized not in VALID_MEMORY_CATEGORIES:
        return "other"

    return normalized


def clamp_score(
    value: float | int | str | None,
    default: float = 0.0,
) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )