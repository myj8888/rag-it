"""Qdrant vector index wrapper.

Input: chunks and query text.
Output: vector search results from one Qdrant collection.
Side effects: calls Ollama embeddings and Qdrant collection APIs.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.rag.embeddings import OllamaEmbeddingClient
from backend.models import Chunk, SearchResult


class DenseIndex:
    def __init__(
        self,
        url: str,
        collection_name: str,
        embedding_client: OllamaEmbeddingClient,
        vector_size: int = 1024,
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding_client = embedding_client
        self.vector_size = vector_size

    def rebuild(self, chunks: list[Chunk], batch_size: int = 16) -> None:
        self._recreate_collection()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedding_client.embed_many([chunk.text for chunk in batch])
            points = [
                qmodels.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "page": chunk.page,
                        "block_type": chunk.block_type,
                        "source_path": chunk.source_path,
                        "image_path": chunk.image_path,
                        "html": chunk.html,
                        "metadata": chunk.metadata,
                    },
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        vector = self.embedding_client.embed(query)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )

        results: list[SearchResult] = []
        for point in points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})
            metadata["image_path"] = payload.get("image_path")
            metadata["html"] = payload.get("html")
            results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    doc_id=str(payload.get("doc_id", "")),
                    text=str(payload.get("text", "")),
                    page=payload.get("page"),
                    block_type=str(payload.get("block_type", "")),
                    source_path=payload.get("source_path"),
                    vector_score=float(point.score or 0.0),
                    metadata=metadata,
                )
            )
        return results

    def count(self) -> int:
        info = self.client.count(collection_name=self.collection_name, exact=True)
        return int(info.count)

    def _recreate_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

