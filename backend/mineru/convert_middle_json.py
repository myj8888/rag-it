"""MinerU middle/layout JSON converter.

Input: MinerU middle/layout JSON with pdf_info pages.
Output: content_list.json-style blocks for RAG ingestion.
Side effects: writes a converted content_list JSON when requested.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def convert_middle_to_content_list(middle_path: Path, output_dir: Path | None = None) -> Path:
    """Convert MinerU middle/layout JSON to a content_list JSON file."""
    with middle_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    blocks = convert_middle_data_to_content_list(data)
    doc_name = _doc_name_from_path(middle_path)
    output_dir = output_dir or middle_path.parent
    output_path = output_dir / f"{doc_name}_content_list.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(blocks, file, ensure_ascii=False, indent=2)

    logger.info("Converted %d MinerU blocks to %s", len(blocks), output_path)
    return output_path


def convert_middle_data_to_content_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return content_list-style blocks from MinerU middle/layout JSON data."""
    blocks: list[dict[str, Any]] = []
    for page in data.get("pdf_info", []):
        page_idx = page.get("page_idx", 0)
        source_blocks = page.get("para_blocks") or page.get("preproc_blocks") or []
        for block in source_blocks:
            converted = _convert_block(block, page_idx)
            if converted is not None:
                blocks.append(converted)
    return blocks


def _doc_name_from_path(path: Path) -> str:
    doc_name = path.stem
    for suffix in ["_content_list", "_middle", "_model", "_span", "_layout"]:
        doc_name = doc_name.removesuffix(suffix)
    if doc_name == "layout":
        doc_name = path.parent.name or "document"
    return doc_name


def _convert_block(block: dict[str, Any], page_idx: int) -> dict[str, Any] | None:
    block_type = block.get("type", "")
    bbox = block.get("bbox")

    if block_type in {"text", "list", "title"}:
        text = _extract_list_text(block) if block_type == "list" else _extract_text(block)
        if not text:
            return None
        result: dict[str, Any] = {
            "type": "text",
            "text": text,
            "bbox": bbox,
            "page_idx": page_idx,
        }
        if block_type == "title":
            result["text_level"] = _infer_heading_level(block)
        return result

    if block_type == "image":
        caption = _extract_nested_text(block, {"image_caption"})
        footnote = _extract_nested_text(block, {"image_footnote"})
        result = {
            "type": "image",
            "img_path": _extract_image_path(block),
            "image_caption": [caption] if caption else [],
            "image_footnote": [footnote] if footnote else [],
            "bbox": bbox,
            "page_idx": page_idx,
        }
        return result

    if block_type == "table":
        return _convert_table_block(block, page_idx, bbox)

    if block_type in {"interline_equation", "equation"}:
        text = _extract_text(block)
        if not text:
            return None
        return {
            "type": "text",
            "text": text,
            "bbox": bbox,
            "page_idx": page_idx,
        }

    return None


def _extract_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []) or []:
        for span in line.get("spans", []) or []:
            content = str(span.get("content", "")).strip()
            if content:
                parts.append(content)
    return " ".join(parts)


def _extract_list_text(block: dict[str, Any]) -> str:
    items = block.get("list_items")
    if isinstance(items, list):
        return "\n".join(str(item).strip() for item in items if str(item).strip())
    return _extract_text(block)


def _extract_nested_text(block: dict[str, Any], block_types: set[str]) -> str:
    parts: list[str] = []
    for nested in block.get("blocks", []) or []:
        if nested.get("type") in block_types:
            text = _extract_text(nested)
            if text:
                parts.append(text)
    return " ".join(parts)


def _extract_image_path(block: dict[str, Any]) -> str:
    for nested in block.get("blocks", []) or []:
        for line in nested.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                image_path = span.get("image_path") or span.get("img_path")
                if image_path:
                    return str(image_path)
    image_path = block.get("image_path") or block.get("img_path")
    return str(image_path or "")


def _extract_html(block: dict[str, Any]) -> str:
    for line in block.get("lines", []) or []:
        for span in line.get("spans", []) or []:
            html = span.get("html")
            if span.get("type") == "table" and html:
                return str(html)
    return ""


def _infer_heading_level(block: dict[str, Any]) -> int:
    bbox = block.get("bbox", [0, 0, 0, 0])
    if isinstance(bbox, list) and len(bbox) >= 4:
        height = bbox[3] - bbox[1]
        if height > 40:
            return 1
        if height > 25:
            return 2
    return 3


def _convert_table_block(
    block: dict[str, Any], page_idx: int, bbox: list[int] | None
) -> dict[str, Any] | None:
    caption = _extract_nested_text(block, {"table_caption"})
    footnote = _extract_nested_text(block, {"table_footnote"})
    html_body = ""

    for nested in block.get("blocks", []) or []:
        if nested.get("type") == "table_body":
            html_body = _extract_html(nested)
            if html_body:
                break

    if not html_body:
        html_body = _extract_html(block)
    if not html_body:
        text = _extract_text(block)
        if not text:
            return None
        html_body = f"<table><tr><td>{text}</td></tr></table>"

    return {
        "type": "table",
        "table_body": html_body,
        "table_caption": [caption] if caption else [],
        "table_footnote": [footnote] if footnote else [],
        "bbox": bbox,
        "page_idx": page_idx,
    }

