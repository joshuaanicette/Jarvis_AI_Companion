from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ModelDecision:
    model: str
    reason: str
    category: str
    requires_reasoning: bool


class ModelRouter:
    REASONING_CATEGORIES = {
        "math",
        "engineering",
        "science",
        "programming",
        "technology",
        "technical",
    }

    CODING_CATEGORIES = {
        "programming",
        "software",
        "software engineering",
        "embedded",
        "embedded systems",
        "c",
        "c++",
        "cpp",
        "firmware",
        "linux",
        "fpga",
    }

    EXPLICIT_QWEN_PHRASES = (
        "use qwen",
        "have qwen",
        "ask qwen",
        "qwen do it",
        "with qwen",
    )

    REASONING_PHRASES = (
        "solve",
        "calculate",
        "derive",
        "prove",
        "analyze",
        "analysis",
        "full breakdown",
        "step by step",
        "show your work",
        "compare",
        "difference between",
        "explain why",
        "complex",
        "advanced",
        "technical",
    )

    CODING_PHRASES = (
        "code",
        "coding",
        "debug",
        "compiler",
        "compile",
        "build error",
        "linker",
        "segmentation fault",
        "unit test",
        "pytest",
        "cmake",
        "makefile",
        "platformio",
        "firmware",
        "embedded",
        "uart",
        "spi",
        "i2c",
        "gpio",
        "interrupt",
        "memory mapped",
        "pointer",
        "c++",
        "cpp",
        "verilog",
        "vitis",
        "vivado",
    )

    COMPLEX_CODING_PHRASES = (
        "refactor",
        "architecture",
        "multi file",
        "multiple files",
        "race condition",
        "memory leak",
        "undefined behavior",
        "linker error",
        "toolchain",
        "driver",
        "device tree",
        "rtos",
        "bare metal",
        "bare-metal",
    )

    def __init__(
        self,
        fast_model: str = "gemma3:1b",
        reasoning_model: str = "qwen2.5:3b",
        coding_model: str = "qwen2.5-coder:3b",
        coding_strong_model: str = "qwen2.5-coder:7b",
    ):
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model
        self.coding_model = coding_model
        self.coding_strong_model = coding_strong_model

    def select_model(
        self,
        text: str,
        category: str = "general",
    ) -> ModelDecision:
        normalized = self._normalize(text)
        normalized_category = str(category).casefold().strip()

        explicit_qwen = any(
            phrase in normalized
            for phrase in self.EXPLICIT_QWEN_PHRASES
        )

        coding_category = normalized_category in self.CODING_CATEGORIES
        coding_request = any(
            phrase in normalized
            for phrase in self.CODING_PHRASES
        )

        complex_coding = any(
            phrase in normalized
            for phrase in self.COMPLEX_CODING_PHRASES
        )

        if coding_category or coding_request:
            if complex_coding:
                return ModelDecision(
                    model=self.coding_strong_model,
                    reason=(
                        "Complex coding/embedded work detected; "
                        "using the stronger local coder."
                    ),
                    category=normalized_category,
                    requires_reasoning=True,
                )

            return ModelDecision(
                model=self.coding_model,
                reason=(
                    "Coding or embedded-systems work detected; "
                    "using the lightweight local coder."
                ),
                category=normalized_category,
                requires_reasoning=True,
            )

        technical_category = normalized_category in self.REASONING_CATEGORIES

        reasoning_request = any(
            phrase in normalized
            for phrase in self.REASONING_PHRASES
        )

        mathematical_notation = bool(
            re.search(
                r"[∫∑√]|"
                r"\b\d*\s*[xyz]\s*\^\s*\d+|"
                r"\bdx\b|"
                r"\bd/dx\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

        circuit_request = any(
            phrase in normalized
            for phrase in (
                "circuit",
                "circuits",
                "phasor",
                "impedance",
                "thevenin",
                "norton",
                "kirchhoff",
                "nodal",
                "mesh analysis",
                "electrical engineering",
                "computer engineering",
            )
        )

        use_reasoning_model = any(
            (
                explicit_qwen,
                technical_category,
                reasoning_request,
                mathematical_notation,
                circuit_request,
            )
        )

        if use_reasoning_model:
            reasons = []

            if explicit_qwen:
                reasons.append("the user explicitly requested Qwen")
            if technical_category:
                reasons.append(f"the subject category is {normalized_category}")
            if reasoning_request:
                reasons.append("the prompt requests detailed reasoning")
            if mathematical_notation:
                reasons.append("mathematical notation was detected")
            if circuit_request:
                reasons.append("engineering or circuit terminology was detected")

            return ModelDecision(
                model=self.reasoning_model,
                reason="; ".join(reasons),
                category=normalized_category,
                requires_reasoning=True,
            )

        return ModelDecision(
            model=self.fast_model,
            reason="The request appears conversational or low-complexity.",
            category=normalized_category,
            requires_reasoning=False,
        )

    def route(self, text: str, category: str = "general") -> ModelDecision:
        return self.select_model(text=text, category=category)

    def choose_model(self, text: str, category: str = "general") -> ModelDecision:
        return self.select_model(text=text, category=category)

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = str(text).casefold()
        normalized = normalized.replace("intergral", "integral")
        normalized = normalized.replace("intergration", "integration")
        return re.sub(r"\s+", " ", normalized).strip()
