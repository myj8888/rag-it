"""Chat LLM client.

Input: chat messages.
Output: model text responses.
Side effects: calls OpenAI-compatible or Ollama chat APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from backend.config import Settings


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(self, messages: list[ChatMessage]) -> str:
        provider = self.settings.llm_provider.lower()
        if provider == "openai_compatible":
            return self._openai_compatible_chat(messages)
        if provider == "ollama":
            return self._ollama_chat(messages)
        raise ValueError(f"Unsupported LLM_PROVIDER: {self.settings.llm_provider}")

    def _openai_compatible_chat(self, messages: list[ChatMessage]) -> str:
        if not self.settings.llm_api_key or self.settings.llm_api_key.startswith("your_"):
            raise RuntimeError("LLM_API_KEY is not configured. Fill it in .env before running rag ask.")
        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.llm_model,
                "messages": [message.__dict__ for message in messages],
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def _ollama_chat(self, messages: list[ChatMessage]) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        response = requests.post(
            url,
            json={
                "model": self.settings.llm_model,
                "messages": [message.__dict__ for message in messages],
                "stream": False,
                "options": {
                    "temperature": self.settings.llm_temperature,
                    "num_predict": self.settings.llm_max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["message"]["content"].strip()

