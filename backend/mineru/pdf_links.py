"""PDF external link extraction.

Input: origin.pdf in a parsed document folder.
Output: PdfLink records with anchor text and basic safety flags.
Side effects: reads PDF files via PyMuPDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz

from backend.mineru.link_safety import assess_link

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfLink:
    uri: str
    page: int
    rect: list[float]
    anchor_text: str
    source_pdf: str
    safety_label: str
    safety_warnings: list[str]


def find_source_pdf(data_dir: Path) -> Path | None:
    preferred = sorted(data_dir.glob("origin.pdf"))
    if preferred:
        return preferred[0]
    preferred = sorted(data_dir.glob("*_origin.pdf"))
    if preferred:
        return preferred[0]
    pdfs = sorted(data_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def extract_pdf_links(data_dir: Path) -> list[PdfLink]:
    pdf_path = find_source_pdf(data_dir)
    if pdf_path is None:
        logger.info("No PDF found for link extraction in %s", data_dir)
        return []

    logger.info("Extracting PDF links from %s", pdf_path)
    links: list[PdfLink] = []
    doc = fitz.open(pdf_path)
    try:
        for page_idx, page in enumerate(doc):
            for raw_link in page.get_links():
                uri = _link_uri(raw_link)
                if not uri:
                    continue
                if not _is_external_link(uri):
                    logger.debug("Skipping internal PDF link: %s", uri)
                    continue
                rect = raw_link.get("from")
                anchor_text = ""
                rect_list: list[float] = []
                if rect is not None:
                    rect_list = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                    expanded = fitz.Rect(rect)
                    expanded.x0 = max(0, expanded.x0 - 80)
                    expanded.x1 = min(page.rect.width, expanded.x1 + 80)
                    expanded.y0 = max(0, expanded.y0 - 8)
                    expanded.y1 = min(page.rect.height, expanded.y1 + 8)
                    anchor_text = page.get_textbox(expanded).strip()
                safety = assess_link(uri)
                links.append(
                    PdfLink(
                        uri=uri,
                        page=page_idx + 1,
                        rect=rect_list,
                        anchor_text=anchor_text,
                        source_pdf=str(pdf_path),
                        safety_label=safety.label,
                        safety_warnings=safety.warnings,
                    )
                )
    finally:
        doc.close()
    logger.info("Extracted %s PDF links", len(links))
    return links


def _link_uri(raw_link: dict) -> str:
    for key in ("uri", "file", "nameddest"):
        value = raw_link.get(key)
        if value:
            return str(value)
    if raw_link.get("page") is not None:
        return f"page:{int(raw_link['page']) + 1}"
    return ""


def _is_external_link(uri: str) -> bool:
    return uri.startswith(("http://", "https://", "mailto:"))

