"""SQLite BM25 sparse index.

Input: chunks and query text.
Output: BM25 search results from one SQLite index file.
Side effects: writes/reads bm25.sqlite3 files.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import jieba

from backend.models import Chunk, SearchResult


class SparseIndex:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    def rebuild(self, chunks: list[Chunk]) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS chunk_fts;
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    block_type TEXT NOT NULL,
                    source_path TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize='unicode61'
                );
                """
            )
            for chunk in chunks:
                metadata = {
                    **chunk.metadata,
                    "image_path": chunk.image_path,
                    "html": chunk.html,
                }
                conn.execute(
                    """
                    INSERT INTO chunks
                    (chunk_id, doc_id, text, page, block_type, source_path, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.page,
                        chunk.block_type,
                        chunk.source_path,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
                conn.execute(
                    "INSERT INTO chunk_fts (chunk_id, text) VALUES (?, ?)",
                    (chunk.chunk_id, tokenize_for_search(chunk.text)),
                )

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        query_expression = fts_query_for_text(query)
        if not query_expression:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.text, c.page, c.block_type, c.source_path,
                       c.metadata_json, bm25(chunk_fts) AS score
                FROM chunk_fts
                JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
                WHERE chunk_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query_expression, top_k),
            ).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            score = float(row["score"])
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    page=row["page"],
                    block_type=row["block_type"],
                    source_path=row["source_path"],
                    bm25_score=1.0 / (1.0 + max(score, 0.0)),
                    metadata=json.loads(row["metadata_json"]),
                )
            )
        return results

    def get_chunk(self, chunk_id: str) -> SearchResult | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT chunk_id, doc_id, text, page, block_type, source_path, metadata_json
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return SearchResult(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            page=row["page"],
            block_type=row["block_type"],
            source_path=row["source_path"],
            metadata=json.loads(row["metadata_json"]),
        )

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn


def tokenize_for_search(text: str) -> str:
    tokens = [
        token.strip()
        for token in jieba.cut_for_search(text)
        if token.strip() and not token.isspace()
    ]
    return " ".join(tokens)


def fts_query_for_text(text: str) -> str:
    stop_words = {"的", "了", "是", "有", "哪些"}
    terms = [token for token in tokenize_for_search(text).split() if token and token not in stop_words]
    if not terms:
        return ""
    return " OR ".join(f'"{term.replace("\"", "\"\"")}"' for term in terms)

