"""Application configuration.

Input: environment variables from .env.
Output: Settings used by CLI, indexing, retrieval, MinerU, and LLM clients.
Side effects: reads .env only; does not call external services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    storage_dir: Path
    indexes_dir: Path
    index_registry_path: Path
    log_level: str
    log_file: Path
    image_ocr_enabled: bool
    pdf_link_extraction_enabled: bool
    qdrant_url: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    ollama_base_url: str
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    bm25_top_k: int
    vector_top_k: int
    final_top_k: int
    rrf_k: int
    qdrant_collection_prefix: str
    mineru_api_token: str
    mineru_model_version: str
    mineru_language: str
    mineru_poll_interval_seconds: int
    mineru_timeout_seconds: int


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    load_dotenv()
    storage_dir = Path(os.getenv("RAG_STORAGE_DIR", "rag_storage"))
    parse_dir = Path(os.getenv("PDF_PARSE_DIR", "pdf_parse"))
    indexes_dir = storage_dir / "indexes"
    return Settings(
        data_dir=parse_dir,
        storage_dir=storage_dir,
        indexes_dir=indexes_dir,
        index_registry_path=storage_dir / "index_registry.json",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=Path(os.getenv("RAG_LOG_FILE", str(storage_dir / "rag_it.log"))),
        image_ocr_enabled=_get_bool("IMAGE_OCR_ENABLED", True),
        pdf_link_extraction_enabled=_get_bool("PDF_LINK_EXTRACTION_ENABLED", True),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
        embedding_dim=_get_int("EMBEDDING_DIM", 1024),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_get_int("LLM_MAX_TOKENS", 2048),
        bm25_top_k=_get_int("BM25_TOP_K", 20),
        vector_top_k=_get_int("VECTOR_TOP_K", 20),
        final_top_k=_get_int("FINAL_TOP_K", 5),
        rrf_k=_get_int("RRF_K", 60),
        qdrant_collection_prefix=os.getenv("QDRANT_COLLECTION_PREFIX", "rag"),
        mineru_api_token=os.getenv("MINERU_API_TOKEN", ""),
        mineru_model_version=os.getenv("MINERU_MODEL_VERSION", "vlm"),
        mineru_language=os.getenv("MINERU_LANGUAGE", "ch"),
        mineru_poll_interval_seconds=_get_int("MINERU_POLL_INTERVAL_SECONDS", 10),
        mineru_timeout_seconds=_get_int("MINERU_TIMEOUT_SECONDS", 3600),
    )

