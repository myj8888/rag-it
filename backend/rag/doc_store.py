"""Parsed document folder helpers.

Input: pdf_parse/<doc_id> folders and manifest.json files.
Output: parsed document records and normalized document ids.
Side effects: may write manifest.json when MinerU parsing finishes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    doc_dir: Path
    content_list_path: Path
    manifest_path: Path | None = None


def safe_doc_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return normalized or "document"


def doc_dir_for_pdf(parse_root: Path, pdf_path: Path) -> Path:
    return parse_root / safe_doc_id(pdf_path.stem)


def iter_parsed_documents(parse_root: Path, doc_id: str | None = None) -> list[ParsedDocument]:
    if not parse_root.exists():
        return []

    doc_dirs = [parse_root / doc_id] if doc_id else sorted(path for path in parse_root.iterdir() if path.is_dir())
    documents: list[ParsedDocument] = []
    for doc_dir in doc_dirs:
        content_list = doc_dir / "content_list.json"
        if not content_list.exists():
            legacy_matches = sorted(doc_dir.glob("*_content_list.json"))
            if legacy_matches:
                content_list = legacy_matches[0]
        if not content_list.exists():
            continue
        manifest_path = doc_dir / "manifest.json"
        documents.append(
            ParsedDocument(
                doc_id=doc_dir.name,
                doc_dir=doc_dir,
                content_list_path=content_list,
                manifest_path=manifest_path if manifest_path.exists() else None,
            )
        )

    if not documents and doc_id is None:
        legacy_matches = sorted(parse_root.glob("*_content_list.json"))
        for content_list in legacy_matches:
            documents.append(
                ParsedDocument(
                    doc_id=content_list.name.removesuffix("_content_list.json"),
                    doc_dir=parse_root,
                    content_list_path=content_list,
                    manifest_path=None,
                )
            )
    return documents


def write_manifest(
    doc_dir: Path,
    doc_id: str,
    origin_pdf: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "doc_id": doc_id,
        "product_name": doc_id,
        "brand": "",
        "model": "",
        "description": "",
        "language": "",
        "version": "",
        "source_url": "",
        "origin_pdf": "origin.pdf",
        "content_list": "content_list.json",
        "middle_json": "middle.json",
        "markdown": "full.md",
        "notes": "Edit product_name/brand/model/description for better routing.",
    }
    if extra:
        manifest.update(extra)
    doc_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = doc_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def read_manifest(doc_dir: Path) -> dict[str, Any]:
    manifest_path = doc_dir / "manifest.json"
    if not manifest_path.exists():
        return {"doc_id": doc_dir.name, "product_name": doc_dir.name}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
