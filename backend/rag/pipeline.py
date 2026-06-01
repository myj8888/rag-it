"""RAG query pipeline.

Input: user query, optional doc_ids, per-document index registry.
Output: search results or generated answer with route information.
Side effects: reads local SQLite indexes, queries Qdrant/Ollama/LLM services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.config import Settings
from backend.models import SearchResult
from backend.rag.dense_index import DenseIndex
from backend.rag.embeddings import OllamaEmbeddingClient
from backend.rag.index_registry import IndexEntry, require_entries
from backend.rag.llm import ChatMessage, LLMClient
from backend.rag.retriever import HybridRetriever, deduplicate_results
from backend.rag.router import RouteDecision, route_query
from backend.rag.sparse_index import SparseIndex


@dataclass(frozen=True)
class RagSearch:
    sources: list[SearchResult]
    route: RouteDecision


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[SearchResult]
    route: RouteDecision


class RagPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )
        self.llm = LLMClient(settings)

    def search(self, query: str, doc_ids: list[str] | None = None) -> RagSearch:
        entries = require_entries(self.settings)
        route = route_query(query, entries, self.settings, doc_ids=doc_ids)
        if not route.selected_doc_ids:
            return RagSearch(sources=[], route=route)

        selected_entries = require_entries(self.settings, route.selected_doc_ids)
        all_results: list[SearchResult] = []
        for entry in selected_entries:
            all_results.extend(self._search_one(entry, query))

        ranked = sorted(all_results, key=lambda result: result.hybrid_score, reverse=True)
        deduped = deduplicate_results(ranked)
        return RagSearch(sources=deduped[: self.settings.final_top_k], route=route)

    def ask(self, query: str, doc_ids: list[str] | None = None) -> RagAnswer:
        search = self.search(query, doc_ids=doc_ids)
        if not search.sources:
            return RagAnswer(
                answer="没有匹配到相关说明书索引，或当前索引中没有检索到足够相关的内容。",
                sources=[],
                route=search.route,
            )
        messages = build_messages(query, search.sources)
        return RagAnswer(answer=self.llm.chat(messages), sources=search.sources, route=search.route)

    def _search_one(self, entry: IndexEntry, query: str) -> list[SearchResult]:
        sparse_index = SparseIndex(Path(entry.sqlite_path))
        dense_index = DenseIndex(
            url=self.settings.qdrant_url,
            collection_name=entry.qdrant_collection,
            embedding_client=self.embedding_client,
            vector_size=self.settings.embedding_dim,
        )
        candidate_top_k = max(self.settings.final_top_k * 3, self.settings.final_top_k + 5)
        retriever = HybridRetriever(
            sparse_index=sparse_index,
            dense_index=dense_index,
            bm25_top_k=self.settings.bm25_top_k,
            vector_top_k=self.settings.vector_top_k,
            final_top_k=candidate_top_k,
            rrf_k=self.settings.rrf_k,
        )
        return retriever.retrieve(query)


def build_messages(query: str, sources: list[SearchResult]) -> list[ChatMessage]:
    context = "\n\n".join(
        f"[来源 {index}] 文档: {source.doc_id}; 页码: {source.page or '未知'}; 类型: {source.block_type}\n{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    system = (
        "你是一个严谨的中文产品说明书 RAG 助手。只能依据用户给出的检索片段回答。"
        "如果片段不足以支持答案，请明确说明当前文档不足以确认。"
        "回答要简洁、可操作，关键结论后标注来源编号，例如[来源 1]。"
        "遇到链接问题时，要提醒用户结合链接安全检测结果谨慎打开。"
    )
    user = f"问题：{query}\n\n检索片段：\n{context}"
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
