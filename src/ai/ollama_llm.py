from typing import Any

import requests

from src.core.logger import logger


class OllamaLLMError(RuntimeError):
    """Raised when Ollama cannot generate a response."""


class OllamaLLM:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "gemma3:1b",
        timeout: float = 180.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
    ) -> str:
        selected_model = model or self.model

        payload: dict[str, Any] = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.4,
                "num_predict": 500,
            },
        }

        if system:
            payload["system"] = system

        logger.info(
            "Sending generation request to Ollama model: %s",
            selected_model,
        )

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

        except requests.RequestException as error:
            raise OllamaLLMError(
                f"Could not connect to Ollama: {error}"
            ) from error

        if response.status_code != 200:
            raise OllamaLLMError(
                f"Ollama returned status "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()
        generated_text = str(
            data.get("response", "")
        ).strip()

        if not generated_text:
            raise OllamaLLMError(
                f"Ollama model {selected_model} "
                "returned an empty response."
            )

        return generated_text

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> str:
        selected_model = model or self.model

        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.4,
                "num_predict": 500,
            },
        }

        logger.info(
            "Sending chat request to Ollama model: %s",
            selected_model,
        )

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )

        except requests.RequestException as error:
            raise OllamaLLMError(
                f"Could not connect to Ollama: {error}"
            ) from error

        if response.status_code != 200:
            raise OllamaLLMError(
                f"Ollama returned status "
                f"{response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        generated_text = str(
            data.get(
                "message",
                {},
            ).get(
                "content",
                "",
            )
        ).strip()

        if not generated_text:
            raise OllamaLLMError(
                f"Ollama model {selected_model} "
                "returned an empty response."
            )

        return generated_text