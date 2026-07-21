from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class SubjectDecision:
    category: str
    confidence: float
    matched_terms: tuple[str, ...] = ()


class SubjectRouter:
    CATEGORY_KEYWORDS = {
        "math": (
            "integral",
            "integrate",
            "integration",
            "derivative",
            "differentiate",
            "calculus",
            "algebra",
            "equation",
            "matrix",
            "vector",
            "vector analysis",
            "complex vector",
            "complex number",
            "gradient",
            "divergence",
            "curl",
            "laplace",
            "fourier",
            "differential equation",
            "probability",
            "statistics",
            "trigonometry",
            "geometry",
            "limit",
            "summation",
        ),

        "engineering": (
            "computer engineering",
            "electrical engineering",
            "mechanical engineering",
            "civil engineering",
            "engineering",
            "circuit",
            "circuits",
            "electrical circuits",
            "circuits 2",
            "circuit analysis",
            "phasor",
            "impedance",
            "admittance",
            "resistor",
            "capacitor",
            "inductor",
            "transistor",
            "op amp",
            "operational amplifier",
            "kirchhoff",
            "thevenin",
            "norton",
            "mesh analysis",
            "nodal analysis",
            "ac circuit",
            "dc circuit",
            "signal processing",
            "control system",
            "microcontroller",
            "embedded system",
            "digital logic",
            "computer architecture",
            "raspberry pi",
            "arduino",
            "robotics",
        ),

        "programming": (
            "python",
            "javascript",
            "java",
            "c++",
            "programming",
            "code",
            "debug",
            "traceback",
            "exception",
            "function",
            "class",
            "algorithm",
            "data structure",
            "api",
            "fastapi",
            "ollama",
            "opencv",
            "sqlite",
            "linux",
            "docker",
        ),

        "science": (
            "physics",
            "chemistry",
            "biology",
            "force",
            "momentum",
            "energy",
            "electric field",
            "magnetic field",
            "electromagnetism",
            "thermodynamics",
            "quantum",
        ),

        "technology": (
            "computer science",
            "artificial intelligence",
            "machine learning",
            "neural network",
            "computer vision",
            "operating system",
            "network",
            "cybersecurity",
            "database",
            "processor",
            "cpu",
            "gpu",
        ),
    }

    REASONING_PHRASES = (
        "solve",
        "calculate",
        "derive",
        "prove",
        "analyze",
        "analysis",
        "compare",
        "difference between",
        "explain why",
        "full breakdown",
        "step by step",
        "show your work",
        "complex",
        "advanced",
        "technical",
        "reasoning",
        "have qwen do it",
        "use qwen",
        "qwen",
    )

    def classify(
        self,
        text: str,
    ) -> SubjectDecision:
        normalized = self._normalize(text)

        scores: dict[str, int] = {
            category: 0
            for category in self.CATEGORY_KEYWORDS
        }

        matches: dict[str, list[str]] = {
            category: []
            for category in self.CATEGORY_KEYWORDS
        }

        for category, keywords in (
            self.CATEGORY_KEYWORDS.items()
        ):
            for keyword in keywords:
                if keyword in normalized:
                    scores[category] += self._keyword_weight(
                        keyword
                    )
                    matches[category].append(keyword)

        # Mathematical notation should force math routing.
        if re.search(
            r"\b\d*\s*[xyz]\s*\^\s*\d+",
            normalized,
        ):
            scores["math"] += 5
            matches["math"].append(
                "polynomial notation"
            )

        if re.search(
            r"[∫∑√]|d/dx|dx\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            scores["math"] += 7
            matches["math"].append(
                "mathematical notation"
            )

        # Circuit quantities strongly imply engineering.
        if re.search(
            r"\b\d+(?:\.\d+)?\s*"
            r"(v|a|ohm|ω|hz|f|h)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            scores["engineering"] += 4
            matches["engineering"].append(
                "engineering units"
            )

        winning_category = max(
            scores,
            key=scores.get,
        )

        winning_score = scores[
            winning_category
        ]

        if winning_score <= 0:
            return SubjectDecision(
                category="general",
                confidence=0.35,
            )

        confidence = min(
            0.99,
            0.55 + winning_score * 0.06,
        )

        return SubjectDecision(
            category=winning_category,
            confidence=confidence,
            matched_terms=tuple(
                matches[winning_category]
            ),
        )

    def route(
        self,
        text: str,
    ) -> SubjectDecision:
        return self.classify(text)

    def classify_question(
        self,
        text: str,
    ) -> SubjectDecision:
        return self.classify(text)

    def requires_reasoning(
        self,
        text: str,
        category: str | None = None,
    ) -> bool:
        normalized = self._normalize(text)

        reasoning_categories = {
            "math",
            "engineering",
            "science",
            "programming",
            "technology",
            "technical",
        }

        if (
            category
            and category.casefold()
            in reasoning_categories
        ):
            return True

        return any(
            phrase in normalized
            for phrase in self.REASONING_PHRASES
        )

    @staticmethod
    def _keyword_weight(
        keyword: str,
    ) -> int:
        # Longer phrases are generally more specific.
        word_count = len(
            keyword.split()
        )

        if word_count >= 3:
            return 5

        if word_count == 2:
            return 3

        return 1

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        normalized = str(text).casefold()

        normalized = normalized.replace(
            "intergral",
            "integral",
        )

        normalized = normalized.replace(
            "intergration",
            "integration",
        )

        normalized = normalized.replace(
            "electical",
            "electrical",
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()