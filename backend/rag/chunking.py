"""Document chunk builder.

Input: DocumentBlock objects from MinerU ingestion.
Output: Chunk objects ready for sparse and dense indexing.
Side effects: none.
"""

from __future__ import annotations

import hashlib

from backend.models import Chunk, DocumentBlock


def build_chunks(
    blocks: list[DocumentBlock], max_chars: int = 900, overlap_chars: int = 120
) -> list[Chunk]:
    chunks: list[Chunk] = []
    headings: dict[int, str] = {}
    images_by_page = _images_by_page(blocks)

    for block in blocks:
        if block.heading_level:
            headings[block.heading_level] = block.text
            for level in list(headings):
                if level > block.heading_level:
                    del headings[level]
            continue

        heading_context = " / ".join(headings[level] for level in sorted(headings))
        if block.block_type == "pdf_link":
            heading_context = ""
        text = block.text
        if heading_context and block.text != heading_context:
            text = f"{heading_context}\n{text}"

        for part_index, text_part in enumerate(_split_text(text, max_chars, overlap_chars)):
            chunk_id = _chunk_id(block.doc_id, block.block_index, part_index, text_part)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=block.doc_id,
                    text=text_part,
                    page=block.page,
                    block_type=block.block_type,
                    source_path=block.source_path,
                    image_path=block.image_path,
                    html=block.html,
                    metadata={
                        "block_index": block.block_index,
                        "part_index": part_index,
                        "bbox": block.bbox,
                        "heading": heading_context,
                        "related_images": _related_images(block, images_by_page),
                    },
                )
            )
    return chunks


def _images_by_page(blocks: list[DocumentBlock]) -> dict[int, list[DocumentBlock]]:
    images: dict[int, list[DocumentBlock]] = {}
    for block in blocks:
        if block.block_type == "image" and block.page is not None and block.image_path:
            images.setdefault(block.page, []).append(block)
    return images


def _related_images(
    block: DocumentBlock,
    images_by_page: dict[int, list[DocumentBlock]],
    max_images: int = 2,
) -> list[str]:
    if block.image_path or block.page is None or not block.bbox:
        return []
    candidates = images_by_page.get(block.page, [])
    if not candidates:
        return []

    ranked = sorted(
        (
            (_image_relation_score(block.bbox, image.bbox), image.image_path)
            for image in candidates
            if image.bbox and image.image_path
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    return [str(image_path) for score, image_path in ranked[:max_images] if image_path and score > 0]


def _image_relation_score(text_bbox: list[float], image_bbox: list[float] | None) -> float:
    if not image_bbox or len(text_bbox) < 4 or len(image_bbox) < 4:
        return 0.0
    text_y0, text_y1 = float(text_bbox[1]), float(text_bbox[3])
    image_y0, image_y1 = float(image_bbox[1]), float(image_bbox[3])
    overlap = max(0.0, min(text_y1, image_y1) - max(text_y0, image_y0))
    text_height = max(1.0, text_y1 - text_y0)
    image_height = max(1.0, image_y1 - image_y0)
    overlap_score = overlap / min(text_height, image_height)

    text_center = (text_y0 + text_y1) / 2
    image_center = (image_y0 + image_y1) / 2
    distance_score = 1.0 / (1.0 + abs(text_center - image_center))
    return overlap_score * 10 + distance_score


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= max_chars:
        return [normalized]

    parts: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            newline = normalized.rfind("\n", start, end)
            punctuation = max(normalized.rfind("。", start, end), normalized.rfind("；", start, end))
            split_at = max(newline, punctuation)
            if split_at > start + max_chars // 2:
                end = split_at + 1
        parts.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(0, end - overlap_chars)
    return [part for part in parts if part]


def _chunk_id(doc_id: str, block_index: int, part_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{doc_id}:{block_index}:{part_index}:{text}".encode("utf-8")).hexdigest()
    return digest[:32]

