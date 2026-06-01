"""Index registry for per-document RAG indexes.

Input: parsed document folders, chunk counts, and storage settings.
Output: index paths, Qdrant collection names, and rag_storage/index_registry.json.
Side effects: reads/writes local registry metadata; does not call external services.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.rag.doc_store import ParsedDocument, read_manifest


@dataclass(frozen=True)
class IndexEntry:
    doc_id: str
    product_name: str
    brand: str
    model: str
    description: str
    parse_dir: str
    sqlite_path: str
    qdrant_collection: str
    chunk_count: int
    updated_at: str


def sqlite_path_for_doc(settings: Settings, doc_id: str) -> Path:
    return settings.indexes_dir / doc_id / "bm25.sqlite3"


def meta_path_for_doc(settings: Settings, doc_id: str) -> Path:
    return settings.indexes_dir / doc_id / "index_meta.json"


def collection_name_for_doc(settings: Settings, doc_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", doc_id).strip("_").lower()
    return f"{settings.qdrant_collection_prefix}_{safe or 'document'}"


def load_registry(registry_path: Path) -> dict[str, IndexEntry]:
    if not registry_path.exists():
        return {}
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return {item["doc_id"]: IndexEntry(**item) for item in data.get("indexes", [])}


def save_registry(registry_path: Path, entries: dict[str, IndexEntry]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indexes": [
            asdict(entry)
            for entry in sorted(entries.values(), key=lambda item: item.doc_id)
        ]
    }
    registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_entry(
    settings: Settings,
    document: ParsedDocument,
    chunk_count: int,
) -> IndexEntry:
    manifest = read_manifest(document.doc_dir)
    entry = IndexEntry(
        doc_id=document.doc_id,
        product_name=str(manifest.get("product_name") or document.doc_id),
        brand=str(manifest.get("brand") or ""),
        model=str(manifest.get("model") or ""),
        description=_description_from_manifest(manifest),
        parse_dir=str(document.doc_dir),
        sqlite_path=str(sqlite_path_for_doc(settings, document.doc_id)),
        qdrant_collection=collection_name_for_doc(settings, document.doc_id),
        chunk_count=chunk_count,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    entries = load_registry(settings.index_registry_path)
    entries[entry.doc_id] = entry
    save_registry(settings.index_registry_path, entries)
    _write_doc_meta(settings, entry)
    return entry


def require_entries(settings: Settings, doc_ids: list[str] | None = None) -> list[IndexEntry]:
    entries = load_registry(settings.index_registry_path)
    if doc_ids:
        missing = [doc_id for doc_id in doc_ids if doc_id not in entries]
        if missing:
            raise FileNotFoundError(f"No built index found for document(s): {', '.join(missing)}")
        return [entries[doc_id] for doc_id in doc_ids]
    return list(entries.values())


def _description_from_manifest(manifest: dict[str, Any]) -> str:
    parts = [
        manifest.get("product_name"),
        manifest.get("brand"),
        manifest.get("model"),
        manifest.get("description"),
        manifest.get("notes"),
    ]
    return " | ".join(str(part).strip() for part in parts if str(part).strip())


def _write_doc_meta(settings: Settings, entry: IndexEntry) -> None:
    meta_path = meta_path_for_doc(settings, entry.doc_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(asdict(entry), ensure_ascii=False, indent=2), encoding="utf-8")
