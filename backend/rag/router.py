"""LLM router for choosing document indexes.

Input: user query and index registry entries.
Output: selected doc_ids plus short routing reasoning.
Side effects: may call the configured LLM; falls back to all docs on errors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from backend.config import Settings
from backend.rag.index_registry import IndexEntry
from backend.rag.llm import ChatMessage, LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteDecision:
    reasoning: str
    selected_doc_ids: list[str]
    used_fallback: bool = False


def route_query(
    query: str,
    entries: list[IndexEntry],
    settings: Settings,
    doc_ids: list[str] | None = None,
) -> RouteDecision:
    if doc_ids:
        return RouteDecision(reasoning="用户手动指定说明书，跳过路由。", selected_doc_ids=doc_ids)
    if not entries:
        return RouteDecision(reasoning="没有已构建的说明书索引。", selected_doc_ids=[])

    try:
        raw = LLMClient(settings).chat(_build_messages(query, entries))
        payload = json.loads(_strip_code_fence(raw))
        selected = [str(doc_id) for doc_id in payload.get("selected_ids", [])]
        available = {entry.doc_id for entry in entries}
        selected = [doc_id for doc_id in selected if doc_id in available]
        reasoning = str(payload.get("reasoning") or "路由完成。")[:80]
        return RouteDecision(reasoning=reasoning, selected_doc_ids=selected)
    except Exception as exc:
        logger.warning("LLM routing failed, falling back to all indexes: %s", exc)
        return RouteDecision(
            reasoning="路由失败，已兜底检索全部已建索引。",
            selected_doc_ids=[entry.doc_id for entry in entries],
            used_fallback=True,
        )


def _build_messages(query: str, entries: list[IndexEntry]) -> list[ChatMessage]:
    knowledge_list = "\n".join(
        f"[ID: {entry.doc_id}] 名称：{entry.product_name} | 品牌：{entry.brand} | 型号：{entry.model} | 描述：{entry.description}"
        for entry in entries
    )
    system = (
        "你是一个专业的 RAG 系统路由助手。你的任务是分析用户的问题，"
        "并判断需要检索哪些知识库才能提供完整、准确的答案。"
        "一个问题可能涉及零个、一个或多个知识库。你必须只输出严格 JSON。"
    )
    user = f"""【可选知识库列表】
{knowledge_list}

【路由规则】
1. 仔细阅读用户的问题，评估回答该问题需要哪些说明书知识。
2. 从列表中挑选出所有相关的知识库 ID。
3. 如果问题与所有知识库都不相关，请返回空列表 []。
4. 只输出纯 JSON，不要使用 Markdown 代码块。

【输出格式要求】
{{
  "reasoning": "限50字以内的简要原因",
  "selected_ids": ["doc_id_1", "doc_id_2"]
}}

用户问题：{query}
请输出 JSON 路由结果："""
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()
