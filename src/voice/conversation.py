from __future__ import annotations

import threading
from typing import Any

from src.core.logger import logger


class ConversationManager:
    def __init__(
        self,
        app,
    ):
        self.app = app

        self._processing_lock = threading.Lock()

        self.history: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
    "You are Jarvis, a local AI companion running on a Raspberry Pi 5. "
    "You are friendly, practical, concise, technically capable, and "
    "naturally funny. Use clever jokes, playful comments, and light "
    "sarcasm when appropriate, but never let humor reduce the accuracy "
    "or clarity of an answer. Do not force a joke into every response. "

    "You have advanced knowledge of technology, artificial intelligence, "
    "computer engineering, electronics, robotics, programming, computer "
    "hardware, operating systems, networking, cybersecurity, and embedded "
    "systems. Provide technically accurate explanations and practical "
    "troubleshooting steps. When helping with code, explain important "
    "logic, identify likely errors, and provide complete working examples "
    "when appropriate. Never invent commands, APIs, components, or test "
    "results. Clearly state when information is uncertain or needs to be "
    "verified. "

    "You can answer advanced mathematics and science questions, including "
    "calculus, differential equations, linear algebra, probability, "
    "statistics, discrete mathematics, physics, chemistry, and engineering "
    "analysis. Solve technical problems carefully and verify the result. "
    "Show the governing formula, substitution, intermediate steps, units, "
    "and final answer when appropriate. Explain why each major step is "
    "valid instead of only giving the answer. Never claim that an integral "
    "is a derivative. Verify antiderivatives by differentiating the final "
    "result. Check calculations, signs, dimensions, assumptions, and units "
    "before presenting a final answer. "

    "You are also highly knowledgeable about anime, manga, Marvel, and DC, "
    "including characters, abilities, storylines, teams, major events, and "
    "adaptations. Distinguish between comic, manga, anime, movie, television, "
    "and alternate-universe continuity. Avoid major spoilers unless the user "
    "requests them or gives permission. For character battles or power-scaling "
    "questions, explain the assumptions, versions, feats, limitations, and "
    "reasoning behind the conclusion. "

    "Adapt the depth of each answer to the question. Be concise for simple "
    "requests and detailed for difficult technical problems. Organize complex "
    "answers clearly, ask a focused question when essential information is "
    "missing, and never pretend to know something you do not know. Use the "
    "provided user memories only when they are accurate, relevant, and helpful."
                )
            }
        ]

        self.max_history_messages = 12

    def process(
        self,
        text: str,
        speak_response: bool = True
    ) -> str:
        cleaned_text = str(text).strip()

        if not cleaned_text:
            return ""

        with self._processing_lock:
            return self._process_locked(
                cleaned_text,
                speak_response=speak_response
            )

    def replace_history(
        self,
        messages: list[dict[str, str]],
    ) -> None:
        """Switch the active model context to one saved browser chat."""

        system_message = self.history[0]
        clean_messages: list[dict[str, str]] = []

        for message in messages[-self.max_history_messages:]:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()

            if role in {"user", "assistant"} and content:
                clean_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        self.history = [system_message, *clean_messages]

    def _process_locked(
        self,
        text: str,
        speak_response: bool = True
    ) -> str:
        logger.info(
            "Processing conversation request"
        )

        tool_response = (
            self.app.tool_router.check_tools(
                text
            )
        )

        if tool_response is not None:
            self._analyze_memory_safely(
                user_text=text,
                allow_interest_inference=False,
            )

            if tool_response:
                if speak_response:
                    self.app.speak(
                        tool_response
                    )

                return tool_response

        category = self._classify_subject(
            text
        )

        model_decision = self._select_model(
            text=text,
            category=category,
        )

        selected_model = self._decision_model(
            model_decision
        )

        logger.info(
            "Selected model %s for category %s",
            selected_model,
            category,
        )

        memory_context = (
            self.app.memory_retriever.get_context(
                text
            )
        )

        messages = self._build_messages(
            user_text=text,
            memory_context=memory_context,
        )

        pause_vision = (
            "qwen"
            in selected_model.lower()
            and getattr(
                self.app.vision,
                "running",
                False,
            )
            and hasattr(
                self.app.vision,
                "pause_event",
            )
        )

        if pause_vision:
            self.app.vision.pause_event.set()

        try:
            response = self._chat(
                messages=messages,
                model=selected_model,
            )

        finally:
            if pause_vision:
                self.app.vision.pause_event.clear()

        response = self._extract_text(
            response
        ).strip()

        if not response:
            response = (
                "I could not generate a response."
            )

        self._append_history(
            role="user",
            content=text,
        )

        self._append_history(
            role="assistant",
            content=response,
        )

        self._analyze_memory_safely(
            user_text=text,
            allow_interest_inference=True,
        )

        if speak_response:
            self.app.speak(
                response
            )

        return response

    def _build_messages(
        self,
        user_text: str,
        memory_context: str,
    ) -> list[dict[str, str]]:
        messages = [
            self.history[0],
        ]

        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": memory_context,
                }
            )

        recent_history = self.history[
            1:
        ][
            -self.max_history_messages:
        ]

        messages.extend(
            recent_history
        )

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        return messages

    def _classify_subject(
        self,
        text: str,
    ) -> str:
        router = getattr(
            self.app,
            "subject_router",
            None,
        )

        if router is None:
            return "general"

        method_names = (
            "classify",
            "route",
            "classify_question",
        )

        for method_name in method_names:
            method = getattr(
                router,
                method_name,
                None,
            )

            if not callable(
                method
            ):
                continue

            try:
                result = method(
                    text
                )
            except Exception as error:
                logger.warning(
                    "Subject classification failed: %s",
                    error,
                )

                return "general"

            if isinstance(
                result,
                str,
            ):
                return result

            if isinstance(
                result,
                dict,
            ):
                category = result.get(
                    "category"
                )

                if category:
                    return str(
                        category
                    )

            category = getattr(
                result,
                "category",
                None,
            )

            if category:
                return str(
                    category
                )

        return "general"

    def _select_model(
        self,
        text: str,
        category: str,
    ) -> Any:
        router = self.app.model_router

        method_names = (
            "select_model",
            "route",
            "choose_model",
        )

        for method_name in method_names:
            method = getattr(
                router,
                method_name,
                None,
            )

            if not callable(
                method
            ):
                continue

            attempts = (
                {
                    "text": text,
                    "category": category,
                },
                {
                    "question": text,
                    "category": category,
                },
                {
                    "text": text,
                },
            )

            for kwargs in attempts:
                try:
                    return method(
                        **kwargs
                    )
                except TypeError:
                    continue

            try:
                return method(
                    text,
                    category,
                )
            except TypeError:
                try:
                    return method(
                        text
                    )
                except TypeError:
                    continue

        reasoning_categories = {
            "engineering",
            "programming",
            "math",
            "science",
            "technology",
            "technical",
            "history",
        }

        if category.lower() in reasoning_categories:
            return getattr(
                router,
                "reasoning_model",
                "qwen2.5:3b",
            )

        return getattr(
            router,
            "fast_model",
            "gemma3:1b",
        )

    def _decision_model(
        self,
        decision: Any,
    ) -> str:
        if isinstance(
            decision,
            str,
        ):
            return decision

        if isinstance(
            decision,
            dict,
        ):
            model = decision.get(
                "model"
            )

            if model:
                return str(
                    model
                )

        model = getattr(
            decision,
            "model",
            None,
        )

        if model:
            return str(
                model
            )

        return "gemma3:1b"

    def _chat(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> Any:
        chat = getattr(
            self.app.llm,
            "chat",
            None,
        )

        if not callable(
            chat
        ):
            raise AttributeError(
                "The configured LLM does not provide chat()."
            )

        try:
            return chat(
                messages=messages,
                model=model,
            )

        except TypeError:
            try:
                return chat(
                    messages,
                    model=model,
                )

            except TypeError:
                return chat(
                    messages
                )

    def _analyze_memory_safely(
        self,
        user_text: str,
        allow_interest_inference: bool,
    ) -> None:
        try:
            relevant = (
                self.app.memory_retriever.retrieve(
                    user_text,
                    limit=10,
                )
            )

            candidates = (
                self.app.memory.get_interest_candidates()
            )

            action = (
                self.app.memory_analyzer.analyze(
                    user_text=user_text,
                    existing_memories=relevant,
                    interest_candidates=candidates,
                )
            )

            if (
                not allow_interest_inference
                and action.get(
                    "action"
                )
                == "reinforce_interest"
            ):
                return

            self._apply_memory_action(
                action=action,
                source_text=user_text,
            )

        except Exception as error:
            logger.warning(
                "Could not analyze conversation memory: %s",
                error,
            )

    def _apply_memory_action(
        self,
        action: dict[str, Any],
        source_text: str,
    ) -> None:
        action_name = action.get(
            "action",
            "ignore",
        )

        if action_name == "ignore":
            return

        if action_name == "save_confirmed":
            saved = (
                self.app.memory.save_confirmed_memory(
                    note=action.get(
                        "note",
                        "",
                    ),
                    category=action.get(
                        "category",
                        "other",
                    ),
                    importance=action.get(
                        "importance",
                        0.75,
                    ),
                    confidence=action.get(
                        "confidence",
                        1.0,
                    ),
                    source="memory_analyzer",
                )
            )

            if saved:
                logger.info(
                    "Saved confirmed memory: %s",
                    saved.get(
                        "note"
                    ),
                )

            return

        if action_name == "reinforce_interest":
            result = (
                self.app.memory.reinforce_interest(
                    topic=action.get(
                        "topic",
                        "",
                    ),
                    category=action.get(
                        "category",
                        "interest",
                    ),
                    evidence_strength=action.get(
                        "evidence_strength",
                        0.55,
                    ),
                    evidence_text=source_text,
                )
            )

            if result:
                logger.info(
                    "Updated memory interest: %s",
                    action.get(
                        "topic"
                    ),
                )

            return

        if action_name == "update_memory":
            updated = (
                self.app.memory.update_memory(
                    memory_id=action.get(
                        "memory_id",
                        "",
                    ),
                    note=action.get(
                        "note"
                    ),
                    category=action.get(
                        "category"
                    ),
                    importance=action.get(
                        "importance"
                    ),
                    confidence=action.get(
                        "confidence"
                    ),
                )
            )

            logger.info(
                "Memory update result: %s",
                updated,
            )

            return

        if action_name == "forget_memory":
            removed = (
                self.app.memory.forget_memory(
                    action.get(
                        "query",
                        "",
                    )
                )
            )

            logger.info(
                "Removed %s matching memories",
                removed,
            )

    def _append_history(
        self,
        role: str,
        content: str,
    ) -> None:
        self.history.append(
            {
                "role": role,
                "content": content,
            }
        )

        system_message = self.history[
            0
        ]

        recent = self.history[
            1:
        ][
            -self.max_history_messages:
        ]

        self.history = [
            system_message,
            *recent,
        ]

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
            response = result.get(
                "response"
            )

            if isinstance(
                response,
                str,
            ):
                return response

            content = result.get(
                "content"
            )

            if isinstance(
                content,
                str,
            ):
                return content

            message = result.get(
                "message"
            )

            if isinstance(
                message,
                dict,
            ):
                message_content = message.get(
                    "content"
                )

                if isinstance(
                    message_content,
                    str,
                ):
                    return message_content

        return str(
            result
        )