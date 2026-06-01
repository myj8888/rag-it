from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.rag.doc_store import ParsedDocument
from backend.rag.index_registry import (
    IndexEntry,
    load_registry,
    upsert_entry,
)
from backend.rag.router import route_query


def test_index_registry_writes_per_doc_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    doc_dir = tmp_path / "pdf_parse" / "canon_cp1300"
    doc_dir.mkdir(parents=True)
    (doc_dir / "manifest.json").write_text(
        json.dumps({"product_name": "Canon SELPHY", "brand": "Canon", "model": "CP1300"}),
        encoding="utf-8",
    )
    document = ParsedDocument(
        doc_id="canon_cp1300",
        doc_dir=doc_dir,
        content_list_path=doc_dir / "content_list.json",
        manifest_path=doc_dir / "manifest.json",
    )

    entry = upsert_entry(settings, document, chunk_count=42)
    registry = load_registry(settings.index_registry_path)

    assert entry.sqlite_path == str(tmp_path / "rag_storage" / "indexes" / "canon_cp1300" / "bm25.sqlite3")
    assert entry.qdrant_collection == "rag_canon_cp1300"
    assert registry["canon_cp1300"].chunk_count == 42
    assert (tmp_path / "rag_storage" / "indexes" / "canon_cp1300" / "index_meta.json").exists()


def test_router_uses_llm_json(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    entry = _index_entry(
        doc_id="canon_cp1300",
        product_name="Canon SELPHY",
        brand="Canon",
        model="CP1300",
        description="photo printer manual",
    )

    class FakeLLM:
        def __init__(self, settings: Settings) -> None:
            pass

        def chat(self, messages: list[Any]) -> str:
            return '{"reasoning": "问题涉及打印机", "selected_ids": ["canon_cp1300"]}'

    monkeypatch.setattr("backend.rag.router.LLMClient", FakeLLM)
    decision = route_query("怎么更换墨盒？", [entry], settings)

    assert decision.selected_doc_ids == ["canon_cp1300"]
    assert decision.used_fallback is False


def test_router_falls_back_to_all_on_bad_json(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    entries = [
        _index_entry(doc_id="a", product_name="A"),
        _index_entry(doc_id="b", product_name="B"),
    ]

    class FakeLLM:
        def __init__(self, settings: Settings) -> None:
            pass

        def chat(self, messages: list[Any]) -> str:
            return "not json"

    monkeypatch.setattr("backend.rag.router.LLMClient", FakeLLM)
    decision = route_query("随便问", entries, settings)

    assert decision.selected_doc_ids == ["a", "b"]
    assert decision.used_fallback is True


def _index_entry(
    doc_id: str,
    product_name: str,
    brand: str = "",
    model: str = "",
    description: str = "",
) -> IndexEntry:
    return IndexEntry(
        doc_id=doc_id,
        product_name=product_name,
        brand=brand,
        model=model,
        description=description,
        parse_dir=f"pdf_parse/{doc_id}",
        sqlite_path=f"rag_storage/indexes/{doc_id}/bm25.sqlite3",
        qdrant_collection=f"rag_{doc_id}",
        chunk_count=0,
        updated_at="2026-05-29T00:00:00+00:00",
    )


def _settings(tmp_path: Path) -> Settings:
    storage_dir = tmp_path / "rag_storage"
    return Settings(
        data_dir=tmp_path / "pdf_parse",
        storage_dir=storage_dir,
        indexes_dir=storage_dir / "indexes",
        index_registry_path=storage_dir / "index_registry.json",
        log_level="INFO",
        log_file=storage_dir / "rag_it.log",
        image_ocr_enabled=False,
        pdf_link_extraction_enabled=False,
        qdrant_url="http://localhost:6333",
        embedding_provider="ollama",
        embedding_model="bge-m3",
        embedding_dim=1024,
        ollama_base_url="http://localhost:11434",
        llm_provider="openai_compatible",
        llm_base_url="https://api.example.com",
        llm_api_key="key",
        llm_model="model",
        llm_temperature=0.2,
        llm_max_tokens=256,
        bm25_top_k=20,
        vector_top_k=20,
        final_top_k=5,
        rrf_k=60,
        qdrant_collection_prefix="rag",
        mineru_api_token="token",
        mineru_model_version="vlm",
        mineru_language="ch",
        mineru_poll_interval_seconds=1,
        mineru_timeout_seconds=10,
    )
