"""Hybrid retriever.

Input: query text plus one sparse index and one dense index.
Output: RRF-fused search results, with helpers for result deduplication.
Side effects: queries SQLite and Qdrant through index wrappers.
"""

from __future__ import annotations

from dataclasses import replace
import re

from backend.rag.dense_index import DenseIndex
from backend.rag.sparse_index import SparseIndex
from backend.models import SearchResult

_URL_PATTERN = re.compile(r"(?:https?://|mailto:)[^\s\])）}>，。；;、]+", re.IGNORECASE)


class HybridRetriever:
    def __init__(
        self,
        sparse_index: SparseIndex,
        dense_index: DenseIndex,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> list[SearchResult]:
        bm25_results = self.sparse_index.search(query, self.bm25_top_k)
        vector_results = self.dense_index.search(query, self.vector_top_k)
        return rrf_fuse(bm25_results, vector_results, self.final_top_k, self.rrf_k)


def rrf_fuse(
    bm25_results: list[SearchResult],
    vector_results: list[SearchResult],
    final_top_k: int,
    rrf_k: int,
) -> list[SearchResult]:
    fused: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}

    for rank, result in enumerate(bm25_results, start=1):
        current = fused.get(result.chunk_id, result)
        fused[result.chunk_id] = replace(current, bm25_score=max(current.bm25_score, result.bm25_score))
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, result in enumerate(vector_results, start=1):
        current = fused.get(result.chunk_id, result)
        fused[result.chunk_id] = replace(
            current,
            vector_score=max(current.vector_score, result.vector_score),
        )
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        replace(fused[chunk_id], hybrid_score=score)
        for chunk_id, score in ranked[:final_top_k]
    ]


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove repeated retrieval hits while preserving the ranked order.

    Link chunks are deduplicated by URL. Other chunks are deduplicated by
    document/page/type plus normalized text, which removes repeated BM25/vector
    hits without hiding genuinely different pages.
    """
    deduped: list[SearchResult] = []
    seen: set[tuple[str, ...]] = set()

    for result in results:
        key = _dedupe_key(result)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)

    return deduped


def _dedupe_key(result: SearchResult) -> tuple[str, ...]:
    if result.block_type == "pdf_link":
        url = _extract_url(result)
        if url:
            return ("pdf_link", result.doc_id, url.lower())

    normalized_text = _normalize_text(result.text)
    return (
        "chunk_text",
        result.doc_id,
        str(result.page or ""),
        result.block_type,
        normalized_text[:240],
    )


def _extract_url(result: SearchResult) -> str | None:
    url = result.metadata.get("url") or result.metadata.get("uri") if result.metadata else None
    if isinstance(url, str) and url.strip():
        return _clean_url(url)

    match = _URL_PATTERN.search(result.text)
    if not match:
        return None
    return _clean_url(match.group(0))


def _clean_url(url: str) -> str:
    return url.strip().rstrip(".,;:!?，。；：！？)]）}>")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

