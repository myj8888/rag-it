"""Shared data models.

Input: parsed document blocks and index payload fields.
Output: typed DocumentBlock, Chunk, and SearchResult objects.
Side effects: none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentBlock:
    doc_id: str
    block_index: int
    block_type: str
    text: str
    page: int | None
    bbox: list[float] | None = None
    source_path: str | None = None
    image_path: str | None = None
    html: str | None = None
    heading_level: int | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int | None
    block_type: str
    source_path: str | None = None
    image_path: str | None = None
    html: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    page: int | None
    block_type: str
    source_path: str | None
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

