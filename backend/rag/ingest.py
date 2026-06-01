"""MinerU parsed output ingestion.

Input: pdf_parse/<doc_id> folders with content_list.json and origin.pdf.
Output: DocumentBlock records for RAG indexing.
Side effects: may run local image OCR and read PDF links.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from backend.rag.doc_store import iter_parsed_documents
from backend.mineru.image_ocr import ocr_image
from backend.models import DocumentBlock
from backend.mineru.pdf_links import extract_pdf_links

logger = logging.getLogger(__name__)


def find_content_list(data_dir: Path) -> Path:
    preferred = data_dir / "content_list.json"
    if preferred.exists():
        return preferred
    candidates = sorted(data_dir.glob("*_content_list.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(p for p in data_dir.glob("*content_list*.json") if "v2" not in p.stem)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No MinerU content list JSON found in {data_dir}")


def load_mineru_blocks(
    data_dir: Path,
    enable_image_ocr: bool = True,
    enable_pdf_links: bool = True,
) -> list[DocumentBlock]:
    content_list_path = find_content_list(data_dir)
    doc_id = _doc_id_for_content_list(content_list_path)
    with content_list_path.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)

    blocks: list[DocumentBlock] = []
    for index, item in enumerate(raw_items):
        block = _parse_item(item, index, doc_id, content_list_path)
        if block is not None:
            blocks.append(block)
            if enable_image_ocr and block.image_path and not _is_remote_path(block.image_path):
                ocr_block = _image_ocr_block(block, len(raw_items) + len(blocks), data_dir)
                if ocr_block is not None:
                    blocks.append(ocr_block)
    if enable_pdf_links:
        blocks.extend(_pdf_link_blocks(data_dir, doc_id, len(raw_items) + len(blocks)))
    logger.info("Loaded %s blocks from %s", len(blocks), data_dir)
    return blocks


def load_parsed_documents(
    parse_root: Path,
    doc_id: str | None = None,
    enable_image_ocr: bool = True,
    enable_pdf_links: bool = True,
) -> list[DocumentBlock]:
    documents = iter_parsed_documents(parse_root, doc_id=doc_id)
    if not documents:
        target = parse_root / doc_id if doc_id else parse_root
        raise FileNotFoundError(f"No parsed documents found in {target}")

    blocks: list[DocumentBlock] = []
    for document in documents:
        blocks.extend(
            load_mineru_blocks(
                document.doc_dir,
                enable_image_ocr=enable_image_ocr,
                enable_pdf_links=enable_pdf_links,
            )
        )
    logger.info("Loaded %s blocks from %s parsed documents", len(blocks), len(documents))
    return blocks


def _parse_item(
    item: dict[str, Any], index: int, doc_id: str, source_path: Path
) -> DocumentBlock | None:
    item_type = item.get("type", "unknown")
    page_idx = item.get("page_idx")
    page = int(page_idx) + 1 if page_idx is not None else None
    bbox = item.get("bbox")

    if item_type == "text":
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        return DocumentBlock(
            doc_id=doc_id,
            block_index=index,
            block_type="text",
            text=text,
            page=page,
            bbox=bbox,
            source_path=str(source_path),
            heading_level=item.get("text_level"),
        )

    if item_type == "table":
        html = str(item.get("table_body", "")).strip()
        caption = _join_texts(item.get("table_caption", []))
        footnote = _join_texts(item.get("table_footnote", []))
        table_text = html_to_text(html)
        parts = [part for part in [caption, table_text, footnote] if part]
        text = "\n".join(parts).strip()
        if not text:
            return None
        return DocumentBlock(
            doc_id=doc_id,
            block_index=index,
            block_type="table",
            text=text,
            page=page,
            bbox=bbox,
            source_path=str(source_path),
            image_path=item.get("img_path"),
            html=html,
        )

    if item_type == "image":
        caption = _join_texts(item.get("image_caption", []))
        footnote = _join_texts(item.get("image_footnote", []))
        text = "\n".join(part for part in [caption, footnote] if part).strip()
        text = text or f"Image: {item.get('img_path', '')}".strip()
        return DocumentBlock(
            doc_id=doc_id,
            block_index=index,
            block_type="image",
            text=text,
            page=page,
            bbox=bbox,
            source_path=str(source_path),
            image_path=item.get("img_path"),
        )

    if item_type in {"equation", "interline_equation", "code", "list"}:
        text = str(item.get("text") or item.get("code_body") or item.get("content") or "").strip()
        if not text and isinstance(item.get("list_items"), list):
            text = "\n".join(str(part).strip() for part in item["list_items"] if str(part).strip())
        if not text:
            return None
        return DocumentBlock(
            doc_id=doc_id,
            block_index=index,
            block_type=item_type,
            text=text,
            page=page,
            bbox=bbox,
            source_path=str(source_path),
        )

    return None


def _join_texts(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def _image_ocr_block(
    image_block: DocumentBlock,
    block_index: int,
    data_dir: Path,
) -> DocumentBlock | None:
    if not image_block.image_path:
        return None
    image_path = data_dir / image_block.image_path
    if not image_path.exists():
        logger.info("Skipping OCR because image file does not exist: %s", image_path)
        return None
    ocr_text = ocr_image(image_path).strip()
    if not ocr_text:
        return None
    return DocumentBlock(
        doc_id=image_block.doc_id,
        block_index=block_index,
        block_type="image_ocr",
        text=f"图片 OCR 文本:\n{ocr_text}",
        page=image_block.page,
        bbox=image_block.bbox,
        source_path=image_block.source_path,
        image_path=image_block.image_path,
    )


def _pdf_link_blocks(data_dir: Path, doc_id: str, start_index: int) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    for offset, link in enumerate(extract_pdf_links(data_dir)):
        anchor = f"\n锚点文本: {link.anchor_text}" if link.anchor_text else ""
        safety = f"\n链接安全检测: {link.safety_label}"
        if link.safety_warnings:
            safety += f"；风险提示: {'；'.join(link.safety_warnings)}"
        blocks.append(
            DocumentBlock(
                doc_id=doc_id,
                block_index=start_index + offset,
                block_type="pdf_link",
                text=f"PDF 跳转链接: {link.uri}{anchor}{safety}",
                page=link.page,
                bbox=link.rect,
                source_path=link.source_pdf,
            )
        )
    return blocks


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[str] = []
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if cells:
            rows.append(" | ".join(cells))
    if rows:
        return "\n".join(rows)
    return soup.get_text(" ", strip=True)


def _doc_id_for_content_list(content_list_path: Path) -> str:
    if content_list_path.name == "content_list.json":
        return content_list_path.parent.name
    return content_list_path.name.removesuffix("_content_list.json")


def _is_remote_path(path: str) -> bool:
    return urlparse(path).scheme.lower() in {"http", "https"}

