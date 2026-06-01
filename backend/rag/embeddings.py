"""Ollama embedding client.

Input: text strings.
Output: embedding vectors.
Side effects: calls the local Ollama HTTP API.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class OllamaEmbeddingClient:
    base_url: str
    model: str
    timeout: int = 120

    def embed(self, text: str) -> list[float]:
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        response = requests.post(
            url,
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Ollama did not return an embedding: {payload}")
        return [float(value) for value in embedding]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

