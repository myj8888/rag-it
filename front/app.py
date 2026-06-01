"""Streamlit frontend.

Input: user query and optional manual document selection.
Output: answer, route decision, and retrieved source chunks.
Side effects: calls backend RAG pipeline and displays local images when available.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from backend.config import load_settings
from backend.logging_config import configure_logging
from backend.rag.index_registry import require_entries
from backend.rag.pipeline import RagPipeline


def _is_remote_path(path: str) -> bool:
    return urlparse(path).scheme.lower() in {"http", "https"}


st.set_page_config(page_title="Product Manual RAG", page_icon="R", layout="wide")
st.title("产品说明书 RAG")

settings = load_settings()
configure_logging(settings.log_level, settings.log_file)

entries = require_entries(settings)
doc_options = ["自动路由"] + [entry.doc_id for entry in entries]

with st.sidebar:
    st.subheader("配置")
    st.text(f"解析目录: {settings.data_dir}")
    st.text(f"索引目录: {settings.indexes_dir}")
    st.text(f"注册表: {settings.index_registry_path}")
    st.text(f"向量库: {settings.qdrant_url}")
    st.text(f"Embedding: {settings.embedding_model}")
    st.text(f"LLM: {settings.llm_model}")
    selected_doc = st.selectbox("检索范围", doc_options)
    st.caption("首次使用请先运行 `uv run rag parse <pdf>`，然后运行 `uv run rag index`。")

query = st.text_input("输入问题", placeholder="例如：怎么更换墨盒？故障灯闪烁是什么意思？")
mode = st.radio("模式", ["问答", "只检索"], horizontal=True)

if st.button("运行", type="primary", disabled=not query.strip()):
    try:
        pipeline = RagPipeline(settings)
        doc_ids = None if selected_doc == "自动路由" else [selected_doc]
        if mode == "只检索":
            search = pipeline.search(query, doc_ids=doc_ids)
            answer = None
            results = search.sources
            route = search.route
        else:
            rag_answer = pipeline.ask(query, doc_ids=doc_ids)
            answer = rag_answer.answer
            results = rag_answer.sources
            route = rag_answer.route

        st.subheader("路由")
        st.write(f"选择的说明书: {', '.join(route.selected_doc_ids) if route.selected_doc_ids else '无'}")
        st.write(f"原因: {route.reasoning}")
        if route.used_fallback:
            st.warning("LLM 路由失败，已兜底检索全部已建索引。")

        if answer is not None:
            st.subheader("答案")
            st.write(answer)

        st.subheader("来源")
        if not results:
            st.info("没有检索到相关片段。")
        for index, result in enumerate(results, start=1):
            title = (
                f"来源 {index} | 文档 {result.doc_id} | 页码 {result.page or '未知'} | "
                f"{result.block_type} | hybrid {result.hybrid_score:.4f}"
            )
            with st.expander(title, expanded=index == 1):
                st.write(result.text)
                image_path = result.metadata.get("image_path")
                if image_path and not _is_remote_path(str(image_path)):
                    base_dir = Path(result.source_path).parent if result.source_path else settings.data_dir
                    absolute_image_path = base_dir / str(image_path)
                    if absolute_image_path.exists():
                        st.image(str(absolute_image_path))
                related_images = result.metadata.get("related_images") or []
                for related_image in related_images:
                    if not related_image or _is_remote_path(str(related_image)):
                        continue
                    base_dir = Path(result.source_path).parent if result.source_path else settings.data_dir
                    absolute_image_path = base_dir / str(related_image)
                    if absolute_image_path.exists():
                        st.image(str(absolute_image_path), caption=f"相关图片: {related_image}")
                st.caption(
                    f"BM25: {result.bm25_score:.4f} | Vector: {result.vector_score:.4f} | "
                    f"chunk_id: {result.chunk_id}"
                )
    except Exception as exc:
        st.error(str(exc))
